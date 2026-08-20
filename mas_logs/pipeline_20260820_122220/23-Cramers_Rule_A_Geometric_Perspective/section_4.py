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

class Section4Scene(TeachingScene):
    def construct(self):
        lines = ["First, calculate the base area A.", "Second, swap column one with b.", "Third, calculate the area of A1.", "Finally, divide A1 area by A area.", "This ratio reveals the first variable value."]
        self.setup_layout("Step-by-Step Visualization", lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        eq1 = MathTex(r"A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}")
        self.place_in_area(eq1, 'B2', 'D3', scale_factor=0.6)
        self.play(Write(eq1))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(ORANGE)
        eq2 = MathTex(r"A_1 = \begin{bmatrix} b_1 & 1 \\ b_2 & 2 \end{bmatrix}").set_color_by_tex("b_1", "#00BFFF").set_color_by_tex("b_2", "#00BFFF")
        self.place_in_area(eq2, 'B4', 'D5', scale_factor=0.8)
        self.play(ReplacementTransform(eq1.copy(), eq2))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(BLUE)
        rect = SurroundingRectangle(eq2, color=BLUE)
        self.play(Create(rect))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN)
        ratio = MathTex(r"x_1 = \frac{\det(A_1)}{\det(A)}")
        self.place_at_grid(ratio, 'D3', scale_factor=1.0)
        self.play(Write(ratio))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        result = MathTex(r"x_1 = 3").set_color(YELLOW)
        self.place_at_grid(result, 'D4', scale_factor=1.0)
        self.play(FadeIn(result))
        self.wait(2)
