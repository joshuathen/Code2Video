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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Vector spaces require two fundamental operations.",
            "Addition and scalar multiplication structure our space.",
            "These operations must satisfy eight core axioms.",
            "Polynomials follow these same algebraic rules.",
            "Different structures obey identical structural laws."
        ]
        self.setup_layout("The Eight Axioms: Rules of the Game", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        axioms_text = VGroup(*[Text(f"Axiom {i+1}", font_size=24, color=WHITE) for i in range(8)])
        axioms_text.arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(axioms_text, 'A1', 'C2', scale_factor=0.7)
        self.play(FadeIn(axioms_text))
        self.lecture[0].set_color("#FF8080")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(Text("Operations", font_size=24, color="#FF8080").next_to(axioms_text, RIGHT)))
        self.lecture[1].set_color("#FF8080")

        # === Animation for Lecture Line 3 ===
        self.play(axioms_text.animate.set_color("#FF8080"))
        self.lecture[2].set_color("#FF8080")

        # === Animation for Lecture Line 4 ===
        poly_example = VGroup(
            MathTex("p(x) = a + bx"),
            MathTex("q(x) = c + dx"),
            MathTex("p+q = (a+c) + (b+d)x")
        ).arrange(DOWN)
        self.place_in_area(poly_example, 'D4', 'F6', scale_factor=0.6)
        self.play(Write(poly_example))
        self.lecture[3].set_color("#80FF80")

        # === Animation for Lecture Line 5 ===
        inverse_visual = VGroup(
            Line(LEFT*0.5, RIGHT*0.5, color="#80FF80"),
            Text("Additive Inverse", font_size=20, color="#80FF80")
        ).arrange(DOWN)
        self.place_at_grid(inverse_visual, 'E4', scale_factor=0.8)
        self.play(Create(inverse_visual))
        self.lecture[4].set_color("#80FF80")
        self.wait(2)
