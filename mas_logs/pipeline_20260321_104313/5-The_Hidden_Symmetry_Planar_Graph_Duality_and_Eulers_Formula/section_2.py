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
        # Configuration for the specific educational topic
        lecture_lines = [
            "- Planar Graph Properties",
            "- Face-Vertex Duality",
            "- Euler's Characteristic: V-E+F=2"
        ]
        
        # Initialize the teaching environment
        self.setup_layout("The Hidden Symmetry: Graph Duality", lecture_lines)

        # Animation sequence demonstrating the grid usage and duality
        # Create a primal face (a square in the grid)
        primal_points = [
            self.grid["B2"], self.grid["B4"], 
            self.grid["D4"], self.grid["D2"]
        ]
        primal_shape = Polygon(*primal_points, color=BLUE, stroke_width=4)
        primal_label = Text("Primal Face", font_size=16, color=BLUE).next_to(primal_shape, UP, buff=0.1)

        # Create a dual vertex at the center of the face
        dual_vertex = Dot(color=RED).move_to(self.grid["C3"])
        dual_label = Text("Dual Vertex", font_size=16, color=RED).next_to(dual_vertex, RIGHT, buff=0.1)

        # Execution
        self.play(Write(self.title))
        self.play(FadeIn(self.lecture, shift=RIGHT))
        self.wait(1)
        
        self.play(Create(primal_shape), Write(primal_label))
        self.play(Flash(self.grid["C3"], color=RED))
        self.play(FadeIn(dual_vertex), Write(dual_label))
        
        self.wait(3)

    def setup_layout(self, title_text, lecture_lines):
        """
        Sets up the blackboard-style layout with a title and sidebar text.
        """
        self.camera.background_color = "#000000"
        
        # Title at the top
        self.title = Text(title_text, font_size=32, color=WHITE).to_edge(UP, buff=0.5)
        self.add(self.title)

        # Sidebar text on the left
        lecture_texts = [Text(line, font_size=24, color=GRAY_A) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        self.lecture.to_edge(LEFT, buff=0.5).shift(UP * 0.5)
        self.add(self.lecture)

        # Coordinate System for the right-hand animation area
        # Rows A-F (Top-Down), Cols 1-6 (Left-Right)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]
        cols = ["1", "2", "3", "4", "5", "6"]

        # Offset the grid to the right side of the screen
        start_x = 1.5
        start_y = 2.0
        spacing = 0.8

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                pos_x = start_x + (j * spacing)
                pos_y = start_y - (i * spacing)
                self.grid[f"{row}{col}"] = np.array([pos_x, pos_y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        """
        Helper to position mobjects precisely within the defined animation grid.
        """
        mobject.scale(scale_factor)
        target_pos = self.grid.get(grid_pos, ORIGIN)
        mobject.move_to(target_pos)
        return mobject
