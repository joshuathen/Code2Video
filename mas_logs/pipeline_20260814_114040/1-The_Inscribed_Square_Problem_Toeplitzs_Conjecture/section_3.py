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
        self.setup_layout("Topological Mapping: Configuration Space", [
            "We map curve point pairs to 3D space.", 
            "A surface represents all possible chords.", 
            "Midpoint and distance conditions reveal the square."
        ])
        
        # Assets
        sq_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg")
        param_space = sq_icon.copy()
        param_space.set_color("#9370DB")
        
        curve_points = VGroup(*[Dot(color="#FFA500") for _ in range(5)]).arrange(RIGHT)
        
        func_f = MathTex("F(s, t) = 0", color="#20B2AA")
        
        boundary = DashedVMobject(Rectangle(width=3.5, height=3.5, color="#FFD700"))

        # === Animation for Lecture Line 1 ===
        self.place_in_area(param_space, 'B2', 'C4', scale_factor=0.6)
        self.play(Create(param_space))
        self.lecture[0].set_color("#9370DB")

        # === Animation for Lecture Line 2 ===
        # Map the square vertices to the curve points
        self.place_in_area(curve_points, 'A5', 'B5', scale_factor=0.5)
        self.play(FadeIn(curve_points))
        self.lecture[1].set_color("#FFA500")
        
        # Define a continuous function F on this space
        self.place_at_grid(func_f, 'B3', scale_factor=0.9)
        self.play(Write(func_f))
        
        # Visualize the boundary of the configuration space
        self.place_in_area(boundary, 'D2', 'E4', scale_factor=0.6)
        self.play(Create(boundary))
        self.lecture[2].set_color("#FFD700")

        self.wait(2)
