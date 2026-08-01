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

class Section4Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # Base Configuration
        self.camera.background_color = "#000000"
        
        # Title Setup
        self.title_mob = Text(title_text, font_size=28, color=WHITE).to_edge(UP, buff=0.5)
        self.add(self.title_mob)

        # Left-side lecture content (bullets)
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture_vgroup = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.3).scale(0.8)
        self.lecture_vgroup.to_edge(LEFT, buff=0.7)
        self.add(self.lecture_vgroup)

        # Define 6x6 animation grid for positioning on the right side
        self.grid_map = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Offset grid to the right side of the screen
                x_coord = 1.0 + (j * 0.8)
                y_coord = 2.0 - (i * 0.8)
                self.grid_map[f"{row}{col}"] = np.array([x_coord, y_coord, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        """Helper to place mobjects on the defined grid."""
        mobject.scale(scale_factor)
        mobject.move_to(self.grid_map[grid_pos])
        return mobject

    def construct(self):
        # Content for the Eigenvectors and Eigenvalues section
        lecture_content = [
            "- Topic: Eigenvectors & Eigenvalues",
            "- Definition: Av = lambda * v",
            "- Eigenvectors: Direction unchanged",
            "- Eigenvalues: Scaling magnitude",
            "- Focus: The Soul of Matrices"
        ]
        
        self.setup_layout("Linear Algebra: Section 4", lecture_content)

        # Create geometric representations
        eigen_circle = Circle(radius=0.35, color=BLUE_C)
        transform_square = Square(side_length=0.6, color=YELLOW_C)
        basis_triangle = Triangle(color=RED_C).scale(0.3)
        
        # Position objects using the grid system
        self.place_at_grid(eigen_circle, "A1")
        self.place_at_grid(transform_square, "B2")
        self.place_at_grid(basis_triangle, "C3")

        # Animation Sequence
        self.play(
            Write(self.title_mob),
            FadeIn(self.lecture_vgroup, shift=RIGHT),
            run_time=1.5
        )
        
        self.play(
            Create(eigen_circle),
            Create(transform_square),
            Create(basis_triangle),
            run_time=2
        )
        
        self.play(
            eigen_circle.animate.shift(RIGHT * 0.5),
            transform_square.animate.rotate(PI / 4),
            run_time=1.5
        )

        self.wait(2)
