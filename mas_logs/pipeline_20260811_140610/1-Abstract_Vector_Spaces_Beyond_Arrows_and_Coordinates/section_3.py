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
        self.setup_layout("Diverse Examples: Functions as Vectors", [
            "Polynomials can behave just like standard vectors.",
            "Function spaces follow the same linear rules.",
            "Traits evolve within a defined vector space."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show polynomial p(x) = ax^2 + bx + c
        poly_formula = MathTex("p(x) = ax^2 + bx + c", color="#F1C40F")
        self.place_at_grid(poly_formula, "B3", scale_factor=0.9)
        self.play(Write(poly_formula))
        self.lecture[0].set_color("#F1C40F")

        # === Animation for Lecture Line 2 ===
        # Show two functions f and g added together
        sum_formula = MathTex("f(x) + g(x) = h(x)", color="#9B59B6")
        self.place_at_grid(sum_formula, "D4", scale_factor=0.8)
        self.play(Write(sum_formula))
        self.lecture[1].set_color("#9B59B6")

        # === Animation for Lecture Line 3 ===
        # Highlight common traits in a list
        traits = VGroup(
            Text("Trait A", color="#1ABC9C"),
            Text("Trait B", color="#1ABC9C"),
            Text("Trait C", color="#1ABC9C")
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(traits, "E3", "F5", scale_factor=0.5)
        self.play(FadeIn(traits))
        self.lecture[2].set_color("#1ABC9C")
        
        self.wait(2)
