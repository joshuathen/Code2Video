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
        self.setup_layout("Mathematical Framework: Calculus of Variations", [
            "We minimize the total travel time integral.",
            "We seek a function, not a single point.",
            "This defines our optimal path curve."
        ])
        
        # === Animation for Lecture Line 1 ===
        # J[y] = integral of L(x, y, y') dx
        functional_formula = MathTex(r"J[y] = \int_{x_0}^{x_1} L(x, y, y') \, dx", color="#7FFF00")
        # Fixed per issue 30
        self.place_in_area(functional_formula, 'A2', 'B5', scale_factor=1.0)
        self.play(FadeIn(functional_formula))
        self.lecture[0].set_color("#7FFF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Abstract space concept
        curve_space = VGroup(*[
            FunctionGraph(lambda x: np.sin(x*i/5), x_range=[-1, 1], color=GRAY).set_opacity(0.3)
            for i in range(-3, 4)
        ])
        # Fixed per issue 31
        self.place_in_area(curve_space, 'D1', 'F2', scale_factor=0.7)
        self.play(Create(curve_space))
        self.lecture[1].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight optimal path y(x) using asset
        # Fixed per issue 32
        optimal_path = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/path.svg")
        optimal_path.set_color("#FFD700")
        self.place_in_area(optimal_path, 'D4', 'F5', scale_factor=0.7)
        self.play(FadeIn(optimal_path))
        self.lecture[2].set_color("#FFD700")
        self.wait(2)
