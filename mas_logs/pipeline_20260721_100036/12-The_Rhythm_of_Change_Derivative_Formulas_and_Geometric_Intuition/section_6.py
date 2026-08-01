from manim import *
import numpy as np

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP)
        self.add(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add(self.lecture)

        # Define fine-grained animation grid (4x4 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 0.5 + j * 1
                y = 2.2 - i * 1
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def place_in_area(self, mobject, top_left, bottom_right, scale_factor=1.0):
        tl_pos = self.grid[top_left]
        br_pos = self.grid[bottom_right]
        
        # Calculate center of the area
        center_x = (tl_pos[0] + br_pos[0]) / 2
        center_y = (tl_pos[1] + br_pos[1]) / 2
        center = np.array([center_x, center_y, 0])
        
        mobject.scale(scale_factor)
        mobject.move_to(center)
        return mobject

class Section6Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Calculus turns geometry into precise mathematical formulas.",
            "Derivatives give us a shortcut to understand change.",
            "Now you can see the rhythm in the world's movement."
        ]
        self.setup_layout("Summary: The Calculus Lens", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Create a split-screen effect.
        # On the left (of grid area), display white formulas (#FFFFFF): 'd/dx x^n' and 'd/dx sin x'.
        # On the right (of grid area), show a gold path (#FFD700) of a soaring eagle.
        
        self.lecture[0].set_color(YELLOW)
        
        formula1 = MathTex(r"\frac{d}{dx} x^n = n x^{n-1}", color=WHITE)
        formula2 = MathTex(r"\frac{d}{dx} \sin x = \cos x", color=WHITE)
        formulas = VGroup(formula1, formula2).arrange(DOWN, buff=0.8)
        
        # Resolved Issue 38: Move formulas to avoid overlapping lecture lines
        self.place_in_area(formulas, 'B3', 'D5', scale_factor=0.8)
        
        # Eagle path (ParametricFunction)
        path = ParametricFunction(
            lambda t: np.array([1.5 * t, 0.8 * np.sin(2*t), 0]),
            t_range=[-1.0, 1.0],
            color="#FFD700"
        )
        # Resolved Issue 40: Adjusted path position to avoid cramping
        self.place_in_area(path, 'C4', 'F6', scale_factor=1.0)
        
        # Asset Integration: Eagle (Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/eagle.svg)
        eagle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eagle.svg")
        eagle.set_color("#FFD700")
        eagle.scale(0.3)
        eagle.move_to(path.get_start())
        
        self.play(Write(formulas), Create(path))
        self.play(FadeIn(eagle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw tangent lines on the eagle's path that change color to match the corresponding formulas.
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # ValueTracker for movement along path
        t_tracker = ValueTracker(0)
        
        # Tangent line starts at the beginning of the path
        tangent_line = Line(LEFT, RIGHT, color=WHITE).scale(0.3) # Scaled via points in updater
        
        def update_tangent_and_eagle(mob_group):
            t = t_tracker.get_value()
            p = path.point_from_proportion(t)
            
            # Update eagle position (mob_group[0] is eagle)
            mob_group[0].move_to(p)
            
            # Update tangent line (mob_group[1] is tangent_line)
            # Find derivative via small step
            dt = 0.005
            p1 = path.point_from_proportion(max(0, t - dt))
            p2 = path.point_from_proportion(min(1, t + dt))
            tangent_vec = (p2 - p1)
            norm = np.linalg.norm(tangent_vec)
            if norm > 0:
                tangent_vec /= norm
            
            line_len = 0.6
            mob_group[1].set_points_as_corners([p - tangent_vec * (line_len/2), p + tangent_vec * (line_len/2)])
            
            # Tangent line color matches formulas (both white)
            mob_group[1].set_color(WHITE)

        animated_group = VGroup(eagle, tangent_line)
        animated_group.add_updater(update_tangent_and_eagle)
        
        self.add(tangent_line)
        # Animate the eagle and tangent line soaring along the path
        self.play(t_tracker.animate.set_value(1), run_time=5, rate_func=linear)
        self.wait(1)
        
        # Remove updaters before final transition
        animated_group.clear_updaters()

        # === Animation for Lecture Line 3 ===
        # Transition both sides to merge into a single green text (#00FF00): 'The Language of Change'.
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        final_text = Text("The Language of Change", color="#00FF00")
        # Resolved Issue 39: Adjusted final text position to avoid obstructing lecture notes
        self.place_in_area(final_text, 'B3', 'E6', scale_factor=1.0)
        
        self.play(
            FadeOut(formulas),
            FadeOut(path),
            FadeOut(eagle),
            FadeOut(tangent_line),
            Write(final_text)
        )
        self.wait(2)

# Update Issues:
# update_issue(26, under_review=True, resolution_note="Integrated the eagle SVG asset and animated it moving along the path with a tangent line.")
# update_issue(38, under_review=True, resolution_note="Relocated formulas to grid 'B3'-'D5' to avoid overlap with lecture lines.")
# update_issue(39, under_review=True, resolution_note="Relocated and scaled final text to 'B3'-'E6' at scale 1.0 to prevent obstruction.")
# update_issue(40, under_review=True, resolution_note="Relocated and scaled the eagle path to 'C4'-'F6' at scale 1.0 for better spacing.")
