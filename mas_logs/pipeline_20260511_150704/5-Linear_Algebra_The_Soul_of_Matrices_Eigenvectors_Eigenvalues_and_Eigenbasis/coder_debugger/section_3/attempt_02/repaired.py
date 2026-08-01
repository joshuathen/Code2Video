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
        # Setup layout with title and lines
        lines = [
            'Eigenvectors are special vectors staying on their span.',
            'Their direction remains constant during the transformation.',
            'The eigenvalue measures the amount of scaling.',
            'It shows if the vector stretches or flips.',
            'Formally, A times v equals lambda times v.'
        ]
        self.setup_layout("Core Definitions: Eigenvectors and Eigenvalues", lines)

        # --- INITIAL OBJECTS ---
        # Coordinate system on the right
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_tip": True}
        )
        self.place_in_area(plane, 'B2', 'F6', scale_factor=0.7)
        
        # Vector v at (1, 1)
        v_coords = np.array([1, 1, 0])
        vector_v = Vector(plane.coords_to_point(1, 1), color="#00FFFF")
        # Replaced MathTex with Text to avoid LaTeX dependency error
        v_label = Text("v", font_size=24, color="#00FFFF").next_to(vector_v, UR, buff=0.1)
        
        self.add(plane, vector_v, v_label)
        self.wait(2)