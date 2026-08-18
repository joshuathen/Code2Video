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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary: The Inverse Relationship", [
            "Differentiation and integration are inverse operations.",
            "They form a closed, powerful logical loop.",
            "Calculus turns infinite sums into algebraic steps."
        ])
        
        # Define icons
        # Note: Storyboard references assets but provides non-existent files. Using MathTex placeholders as original code.
        diff_icon = MathTex(r"\\frac{d}{dx}", color="#FF9900", font_size=72)
        int_icon = MathTex(r"\\int", color="#00CCFF", font_size=72)
        formula_group = VGroup(diff_icon, int_icon).arrange(RIGHT, buff=0.5)
        
        # Define loop
        arc_path = Arc(radius=0.7, start_angle=0, angle=PI, color=WHITE)
        sum_algebra_label = Text("Sum -> Algebra", font_size=24, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        # Differentiation and integration are inverse operations.
        self.play(self.lecture[0].animate.set_color("#FF9900"))
        self.place_in_area(formula_group, 'C3', 'D4', scale_factor=1.2)
        self.play(Write(diff_icon), Write(int_icon))

        # === Animation for Lecture Line 2 ===
        # They form a closed, powerful logical loop.
        self.play(self.lecture[1].animate.set_color("#00CCFF"))
        self.place_in_area(arc_path, 'C3', 'C5', scale_factor=0.85)
        self.play(Create(arc_path))
        
        # === Animation for Lecture Line 3 ===
        # Calculus turns infinite sums into algebraic steps.
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.place_at_grid(sum_algebra_label, 'E4', scale_factor=0.9)
        self.play(FadeIn(sum_algebra_label))
        self.wait(2)
