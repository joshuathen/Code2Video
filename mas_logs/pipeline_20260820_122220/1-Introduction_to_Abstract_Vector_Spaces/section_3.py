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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Non-Geometric Vectors", [
            "Vectors can be polynomials or functions.", 
            "Represent a polynomial as a vector.", 
            "Coefficients map directly to vector components."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        # Though the asset is generic, including it as a visual element next to the text.
        vec_text = MathTex(r"v = [1, 2, 3]^T", color=WHITE)
        self.place_in_area(vec_text, 'B4', 'B6', scale_factor=0.9)
        self.play(Write(vec_text))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        # Draw box around list representing vector space color #00FFFF
        box = SurroundingRectangle(vec_text, color="#00FFFF", buff=0.2)
        self.play(Create(box))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Highlight each item showing they act like vectors color #FFFF00
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg]
        poly_text = MathTex(r"P(x) = 1x^2 + 2x + 3", color="#FFFF00")
        self.place_in_area(poly_text, 'D4', 'D6', scale_factor=0.9)
        self.play(Write(poly_text))
        self.lecture[2].set_color("#FFFF00")
        
        self.wait(2)
