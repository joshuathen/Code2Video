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
        self.setup_layout("Visualizing the 2x2 Case", [
            "The 2x2 determinant formula is ad minus bc.",
            "'a' and 'd' stretch the sides.",
            "'b' and 'c' create a shear effect."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Fade in matrix
        matrix = MathTex(r"A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}", font_size=48)
        self.place_in_area(matrix, 'A4', 'C6', scale_factor=0.9)
        self.play(FadeIn(matrix))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#2ECC71")
        # Highlight a and d using get_part_by_tex for robustness
        a_rect = SurroundingRectangle(matrix.get_part_by_tex("a"), color="#2ECC71")
        d_rect = SurroundingRectangle(matrix.get_part_by_tex("d"), color="#2ECC71")
        self.play(Create(a_rect), Create(d_rect))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#E74C3C")
        # Highlight b and c using get_part_by_tex
        b_rect = SurroundingRectangle(matrix.get_part_by_tex("b"), color="#E74C3C")
        c_rect = SurroundingRectangle(matrix.get_part_by_tex("c"), color="#E74C3C")
        self.play(ReplacementTransform(a_rect, b_rect), ReplacementTransform(d_rect, c_rect))
        
        # Show formula result
        formula = MathTex(r"\det(A) = ad - bc", font_size=48)
        self.place_in_area(formula, 'D4', 'F6', scale_factor=0.85)
        formula.set_color("#F1C40F")
        self.play(Write(formula))
        
        self.wait(2)
