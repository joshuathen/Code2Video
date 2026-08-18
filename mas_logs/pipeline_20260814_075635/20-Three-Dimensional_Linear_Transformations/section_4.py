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
        self.setup_layout("Visualizing Determinants and Volume", [
            "The determinant measures volume scaling.",
            "A zero determinant flattens space.",
            "Negative values signify a space inversion."
        ])
        self.add(self.lecture)
        
        # Load asset
        cube_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg"
        cube = SVGMobject(cube_asset, fill_opacity=0.6, fill_color=WHITE)
        
        # Initial placement of lecture text
        self.place_in_area(self.lecture, 'A1', 'C3', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.place_in_area(cube, 'A4', 'F6', scale_factor=0.6)
        self.play(Create(cube))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF5733")
        flattened_cube = cube.copy()
        # Squash transform
        flattened_cube.apply_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]]))
        self.play(Transform(cube, flattened_cube, run_time=2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#33FF57")
        inverted_cube = cube.copy()
        # Inversion transform
        inverted_cube.apply_matrix(np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]]))
        self.play(Transform(cube, inverted_cube, run_time=2))
        self.wait(1)
