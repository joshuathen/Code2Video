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
            "3D objects project shadows onto 2D surfaces.",
            "Similarly, 4D objects project shadows into 3D space.",
            "We visualize these via rotating wireframe projections."
        ]
        self.setup_layout("Projecting Higher Dimensions", lecture_lines)
        
        # Define objects
        # Using SVGMobject as per requirement for svg assets
        cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg", color=WHITE)
        cube_label = Text("Cube vertices", font_size=24, color=WHITE)
        
        shadow_rect = Rectangle(width=1.5, height=1.5, color="#00FF00", fill_opacity=0.5)
        shadow_label = Text("2D Shadow", font_size=24, color="#00FF00")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.place_at_grid(cube, 'B4', scale_factor=0.8)
        self.play(Create(cube), Write(self.place_at_grid(cube_label, 'B5', scale_factor=0.6)))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#00FF00")
        self.place_at_grid(shadow_rect, 'D4', scale_factor=0.8)
        self.play(FadeIn(shadow_rect), Write(self.place_at_grid(shadow_label, 'D5', scale_factor=0.6)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#00FFFF")
        self.play(Rotate(cube, angle=PI/4, axis=OUT), Rotate(cube, angle=PI/4, axis=UP))
        self.wait(2)
