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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Vectors are arrows with magnitude and direction.",
            "Visualize them starting from the origin.",
            "A bird flies from nest at origin.",
            "To a tree at point (3,2).",
            "This arrow represents our vector (3,2)."
        ]
        self.setup_layout("What is a Vector? (Visualizing Magnitude & Direction)", lecture_lines)
        
        # Grid/Axes
        axes = Axes(x_range=[0, 5, 1], y_range=[0, 4, 1], axis_config={"color": "#404040"}).scale(0.5)
        self.place_in_area(axes, 'B3', 'E6', scale_factor=0.9)
        self.add(axes)

        # Assets
        bird = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bird.svg")
        nest = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/nest.svg")
        
        self.place_at_grid(nest, 'E3', scale_factor=0.3)
        self.add(nest)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        vector = Arrow(start=axes.c2p(0, 0), end=axes.c2p(3, 2), color="#FFD700")
        
        # Place bird at tip (3,2)
        bird.move_to(axes.c2p(3, 2)).scale(0.3)
        self.play(Create(vector), FadeIn(bird))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFD700")
        coordinate_label = MathTex("(3,2)", font_size=24, color=WHITE)
        self.place_at_grid(coordinate_label, 'D5', scale_factor=0.7)
        self.play(Write(coordinate_label))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00BFFF")
        vector_formula = MathTex("v = (3,2)", color="#00BFFF")
        self.place_at_grid(vector_formula, 'B5', scale_factor=0.8)
        self.play(Write(vector_formula))
        self.wait(2)
