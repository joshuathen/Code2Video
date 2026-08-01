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

        # Define fine-grained animation grid (6x6 grid on right side)
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
        # Setup layout with the specific title and lecture lines
        lecture_lines_text = [
            'We find eigenvalues using the characteristic equation.',
            "Subtract lambda from the matrix's diagonal entries.",
            'Set the determinant of this matrix to zero.',
            'This creates a polynomial equation to solve.',
            'The roots of this polynomial are our eigenvalues.'
        ]
        self.setup_layout("The Math Behind the Magic: Characteristic Equation", lecture_lines_text)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        # Using raw strings for LaTeX to ensure correct character processing
        eq1 = MathTex(r"A\mathbf{v} = \lambda \mathbf{v}", color=WHITE)
        eq2 = MathTex(r"(A - \lambda I)\mathbf{v} = \mathbf{0}", color=WHITE)
        
        self.place_at_grid(eq1, "A3", scale_factor=0.8)
        self.play(Write(eq1))
        self.wait(1)
        
        # Match scale and position for transform
        eq2.scale(0.8).move_to(eq1.get_center())
        self.play(Transform(eq1, eq2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        char_eq = MathTex(r"\det(A - \lambda I) = 0", color=YELLOW)
        matrix_a = MathTex(r"A = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}", color=WHITE)
        sub_lambda = MathTex(r"A - \lambda I = \begin{bmatrix} 2 - \lambda & 1 \\ 1 & 2 - \lambda \end{bmatrix}", color=WHITE)
        
        # Organizing the setup components
        matrix_setup_group = VGroup(char_eq, matrix_a, sub_lambda).arrange(DOWN, buff=0.4)
        self.place_in_area(matrix_setup_group, 'A2', 'B6', scale_factor=0.8)
        
        # Transition from general equation to specific matrix setup
        self.play(FadeOut(eq1))
        self.play(Write(char_eq))
        self.play(FadeIn(matrix_a))
        self.play(Transform(matrix_a, sub_lambda))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.play(Indicate(char_eq))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        # Step-by-step polynomial calculation
        calc_step1 = MathTex(r"(2 - \lambda)(2 - \lambda) - (1)(1) = 0", color=WHITE)
        calc_step2 = MathTex(r"\lambda^2 - 4\lambda + 3 = 0", color=WHITE)
        calc_step3 = MathTex(r"(\lambda - 3)(\lambda - 1) = 0", color=WHITE)
        
        polynomial_calc = VGroup(calc_step1, calc_step2, calc_step3).arrange(DOWN, buff=0.3)
        self.place_in_area(polynomial_calc, 'C2', 'E6', scale_factor=0.7)
        
        self.play(Write(calc_step1))
        self.wait(0.5)
        self.play(Write(calc_step2))
        self.wait(0.5)
        self.play(Write(calc_step3))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GREEN)
        # Final roots/eigenvalues in green
        final_eigenvalues = MathTex(r"\lambda_1 = 3, \lambda_2 = 1", color=GREEN)
        self.place_at_grid(final_eigenvalues, 'F4', scale_factor=0.9)
        
        self.play(Write(final_eigenvalues))
        self.wait(2)