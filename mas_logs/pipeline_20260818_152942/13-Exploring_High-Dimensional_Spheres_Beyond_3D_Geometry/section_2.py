from manim import *

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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["High-dimensional volume behaves in counter-intuitive ways.", "N-ball volumes concentrate near the outer shell.", "The hyper-orange's mass exists mostly in the peel."]
        self.setup_layout("The Geometry of High-Dimensional Volume", lecture_lines)
        
        # Load asset
        orange_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/orange.svg")
        
        # === Animation for Lecture Line 1 ===
        # Create a unit hypercube in high-D space, color #FF00FF.
        hypercube = Square(color="#FF00FF", fill_opacity=0.3)
        # Using the fix from issue 25 (overrides previous placement suggestions)
        self.place_at_grid(hypercube, 'D5', scale_factor=1.0)
        
        orange_ref = orange_icon.copy()
        orange_ref.scale(0.5).next_to(hypercube, UP)
        
        self.play(FadeIn(hypercube), FadeIn(orange_ref))
        self.play(self.lecture[0].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 2 ===
        # Show volume calculation formula: V = s^n, color #FFFF00.
        formula = MathTex(r"V = s^n", color="#FFFF00")
        # Using the fix from issue 24
        self.place_at_grid(formula, 'B4', scale_factor=1.0)
        
        self.play(Write(formula))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))

        # === Animation for Lecture Line 3 ===
        # Animate hypercube stretching, illustrating volume changes, color #00FFFF.
        stretch_factor = ValueTracker(1.0)
        
        # hypercube.add_updater needs to update scale. 
        # Since the hypercube is a square, we use stretch.
        hypercube.add_updater(lambda mob: mob.set_width(1.0 * stretch_factor.get_value()))
        
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.play(FadeIn(orange_ref.copy().next_to(hypercube, DOWN)))
        self.play(stretch_factor.animate.set_value(2.0), run_time=2)
        self.play(stretch_factor.animate.set_value(1.0), run_time=2)
        
        self.wait(2)
