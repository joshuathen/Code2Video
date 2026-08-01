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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initialization with synced storyboard content
        title_text = "The Computational Shortcut: Diagonalization"
        lecture_lines = [
            "Infinite sums are hard to calculate directly.",
            "For diagonalizable matrices, we use the Eigen-shortcut.",
            "Simply exponentiate the diagonal entries of D."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight current line
        self.play(self.lecture[0].animate.set_color("#66CCFF"))
        
        # Formula: Infinite sum for matrix exponential
        # Fixed: Changed MathTex to Text to avoid latex dependency error
        sum_formula = Text("e^A = sum( A^n / n! )", color=WHITE)
        # Positioning formula in top area
        self.place_in_area(sum_formula, "A1", "B6", scale_factor=0.8)
        
        self.play(Write(sum_formula))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Switch highlight to second line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#66CCFF")
        )
        
        # Formula: Diagonalization decomposition
        # Fixed: Changed MathTex to Text to avoid latex dependency error
        diag_shortcut = Text("A = P D P^-1", color=WHITE)
        # Positioning formula in middle area
        self.place_in_area(diag_shortcut, "C1", "D6", scale_factor=0.85)
        
        self.play(Write(diag_shortcut))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Switch highlight to third line (Yellow highlight to match e^D color)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        
        # Matrix: Exponentiated diagonal entries
        # Fixed: Changed MathTex to Text to avoid latex dependency error
        exp_d_matrix = Text(
            "e^D = [[exp(L1), 0], [0, exp(Ln)]]",
            color="#FFFF00"
        )
        # Positioning matrix in bottom area
        self.place_in_area(exp_d_matrix, "E1", "F6", scale_factor=0.7)
        
        self.play(Write(exp_d_matrix))
        self.wait(1.5)
        
        # Final reconstruction e^A = P e^D P^{-1}
        # Fixed: Changed MathTex to Text to avoid latex dependency error
        reconstruction = Text("e^A = P e^D P^-1", color=WHITE)
        # Position reconstruction clearly in the middle, replacing the intermediate decomposition
        self.play(FadeOut(diag_shortcut), FadeOut(sum_formula))
        self.place_in_area(reconstruction, "B1", "D6", scale_factor=0.9)
        self.play(Write(reconstruction))
        self.wait(2.5)

        # Final cleanup: remove highlight
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
