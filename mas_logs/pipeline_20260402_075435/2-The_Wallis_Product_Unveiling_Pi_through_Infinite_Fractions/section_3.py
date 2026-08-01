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

class Section3Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=32, color=WHITE).to_edge(UP, buff=0.5)
        self.add(self.title)

        # Left-side lecture content
        lecture_texts = [Text(line, font_size=24, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        self.lecture.to_edge(LEFT, buff=0.8)
        self.add(self.lecture)

        # Grid system for right-side animation area
        # Mapping 4x4 grid coordinates to the right side of the screen
        self.grid = {}
        rows = ["A", "B", "C", "D"]
        cols = ["1", "2", "3", "4"]
        
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Offset x to the right half, distribute y vertically
                x_pos = 1.5 + j * 1.2
                y_pos = 1.5 - i * 1.2
                self.grid[f"{row}{col}"] = np.array([x_pos, y_pos, 0])

    def get_pos(self, grid_id):
        return self.grid.get(grid_id, ORIGIN)
