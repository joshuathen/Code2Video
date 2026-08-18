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
        self.setup_layout("Summary & Synthesis", [
            "Vector spaces are sets with algebraic structure.",
            "Arrows, matrices, and functions are all examples.",
            "Abstraction unifies diverse mathematical concepts."
        ])
        
        # === Animation for Lecture Line 1 ===
        vs_text = Text("Vector Space", font_size=40, color="#FFFFFF")
        self.place_in_area(vs_text, "B2", "E5", scale_factor=0.8)
        self.play(Write(vs_text))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/arrow.svg]
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg]
        arrow = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arrow.svg", color="#E74C3C")
        matrix = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/matrix.svg", color="#3498DB")
        func = MathTex(r"f(x) = x^2", color="#2ECC71")
        
        self.place_at_grid(arrow, "A3", scale_factor=0.7)
        self.place_at_grid(matrix, "B4", scale_factor=0.6)
        self.place_at_grid(func, "D4", scale_factor=0.6)
        
        self.play(FadeIn(arrow), FadeIn(matrix), FadeIn(func))
        self.lecture[1].set_color("#E74C3C")

        # === Animation for Lecture Line 3 ===
        abs_text = Text("Abstraction", font_size=48, color="#F39C12")
        self.place_in_area(abs_text, "C2", "D5", scale_factor=0.8)
        
        self.play(FadeOut(vs_text), FadeOut(arrow), FadeOut(matrix), FadeOut(func))
        self.play(GrowFromCenter(abs_text))
        self.play(abs_text.animate.set_color(WHITE), run_time=1)
        self.lecture[2].set_color("#F39C12")
        self.wait(1)
