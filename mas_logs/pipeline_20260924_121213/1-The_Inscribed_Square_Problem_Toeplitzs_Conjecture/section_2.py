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
        lecture_lines = ["Can we fit a square inside any loop?", "Visualize a blob transforming shape.", "Does a square always exist?"]
        self.setup_layout("Defining the Problem: The Inscribed Square", lecture_lines)
        
        # Assets
        loop = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blob.svg")
        loop.set_color("#00FFFF")
        loop_label = Text("Loop", font_size=20, color="#00FFFF")
        
        square = Square(color="#FFFF00", side_length=0.8)
        square_label = Text("Square", font_size=20, color=WHITE)
        
        vertices = VGroup(*[Dot(color="#FF0000", radius=0.08) for _ in range(4)])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.place_in_area(loop, 'B2', 'D4', scale_factor=1.0)
        self.place_at_grid(loop_label, 'B2', scale_factor=0.8)
        self.play(FadeIn(loop), Write(loop_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        self.place_in_area(square, 'C3', 'D4', scale_factor=0.8)
        self.place_at_grid(square_label, 'D5', scale_factor=0.8)
        self.play(FadeIn(square), Write(square_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        # Position vertices on corners of square
        for i, vertex in enumerate(vertices):
            vertex.move_to(square.get_vertices()[i])
        self.play(Create(vertices))
        self.play(square.animate.rotate(PI/4), vertices.animate.rotate(PI/4, about_point=square.get_center()), run_time=2)
        self.wait(2)
