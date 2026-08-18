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
        self.setup_layout("Linear Transformations and Orientation", ["Cross products define linear transformations.", "They map vectors to a new space.", "This preserves coordinate system handedness."])
        
        # Load Assets
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        vector_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg")
        
        # Initialize Transformable Group
        transform_group = VGroup(grid_asset, vector_asset)
        # Fix for issue 35/37: Place within right side grid, maintain gutter
        self.place_in_area(transform_group, "B4", "E6", scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(transform_group))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Transform basis
        matrix = [[1, 1], [0, 1]]
        self.play(ApplyMatrix(matrix, transform_group))
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(RED)

        # === Animation for Lecture Line 3 ===
        # Show Handedness label
        label = Text("Preserved Handedness", font_size=20, color=GREEN)
        # Fix for issue 36: Move label to E4
        self.place_at_grid(label, "E4", scale_factor=0.7)
        self.play(Write(label))
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        self.wait(2)
