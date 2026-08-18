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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Two points define a secant line.",
            "Shrink the distance between them now.",
            "The secant transforms toward the tangent.",
            "The limit reaches the instantaneous slope.",
            "Calculus captures motion at any instant."
        ]
        self.setup_layout("The Process: The Limit of Secants", lecture_lines)
        
        # Setup Plot
        axes = Axes(x_range=[-1, 5], y_range=[-1, 5], axis_config={"include_tip": False})
        self.place_in_area(axes, 'C2', 'F6', scale_factor=0.5)
        curve = axes.plot(lambda x: 0.2 * x**2 + 1, color=WHITE)
        
        p1 = axes.c2p(1, 1.2)
        dot1 = Dot(p1, color=WHITE)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        # Note: icon is empty/placeholder as per path, ignored as per visual instructions.
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(axes), Create(curve), FadeIn(dot1))
        
        p2_tracker = ValueTracker(3)
        
        # Setup persistent objects
        line_group = VGroup()
        dot2 = Dot(axes.c2p(3, 0.2 * 3**2 + 1), color="#00FFFF")
        self.place_at_grid(dot2, 'D4', scale_factor=0.6)
        
        def update_secant(mob):
            x1, x2 = 1, p2_tracker.get_value()
            y1, y2 = 0.2*x1**2 + 1, 0.2*x2**2 + 1
            new_line = Line(axes.c2p(x1, y1), axes.c2p(x2, y2), color="#00FFFF")
            mob.become(new_line)
            
        secant_line = Line(axes.c2p(1, 1.2), axes.c2p(3, 2.8), color="#00FFFF")
        secant_line.add_updater(update_secant)
        
        self.add(secant_line, dot2)
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        self.play(p2_tracker.animate.set_value(1.1), run_time=2)
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        tangent = Line(axes.c2p(1, 1.2), axes.c2p(1.1, 1.24), color="#FF0000") # Simple tangent
        self.play(ReplacementTransform(secant_line, tangent), FadeOut(dot2))
        self.lecture[2].set_color("#FF0000")

        # === Animation for Lecture Line 4 ===
        self.play(FadeOut(tangent))
        self.lecture[3].set_color("#FF0000")

        # === Animation for Lecture Line 5 ===
        self.play(FadeOut(axes), FadeOut(curve), FadeOut(dot1))
        self.lecture[4].set_color(YELLOW)
