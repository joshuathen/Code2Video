from manim import *
import numpy as np

class Section2Scene(Scene):
    def construct(self):
        # 1. Setup Layout
        title_text = "Linear Algebra: Eigenvectors & Eigenvalues"
        lecture_lines = [
            "- Determinants & Scaling",
            "- Linear Transformations",
            "- Defining Eigenvectors",
            "- Finding Eigenvalues",
            "- The Characteristic Equation"
        ]
        self.setup_layout(title_text, lecture_lines)

        # 2. Demonstration - Vector Transformation
        # Create a coordinate system for the right side
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            background_line_style={
                "stroke_color": TEAL,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            }
        )
        
        # Move plane to a grid position (roughly C4/D4 area)
        plane_pos = self.grid["C4"]
        plane.move_to(plane_pos)
        
        # Define a vector and a matrix
        # More precise vector placement using plane's coordinate system
        v = Vector(plane.coords_to_point(1, 1) - plane.coords_to_point(0, 0), color=YELLOW).shift(plane.coords_to_point(0, 0))
        
        label = MathTex("\\vec{v}", color=YELLOW).next_to(v, RIGHT, buff=0.1)
        
        self.play(Create(plane), run_time=1)
        self.play(GrowArrow(v), Write(label))
        self.wait(1)

        # Transformation: A = [[2, 0], [0, 1]] -> v becomes [2, 1]
        v_transformed = Vector(plane.coords_to_point(2, 1) - plane.coords_to_point(0, 0), color=PINK).shift(plane.coords_to_point(0, 0))
        label_transformed = MathTex("A\\vec{v}", color=PINK).next_to(v_transformed, RIGHT, buff=0.1)

        self.play(
            Transform(v, v_transformed),
            Transform(label, label_transformed),
            run_time=2
        )
        self.wait(2)

    def setup_layout(self, title_text, lecture_lines):
        # Camera background
        self.camera.background_color = "#000000"

        # Title at the top
        self.title = Text(title_text, font_size=32, color=WHITE).to_edge(UP, buff=0.5)
        self.add(self.title)

        # Left-side lecture content
        lecture_vgroup = VGroup()
        for line in lecture_lines:
            text = Text(line, font_size=20, color=LIGHT_GRAY)
            lecture_vgroup.add(text)
        
        lecture_vgroup.arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        lecture_vgroup.to_edge(LEFT, buff=0.8).shift(DOWN * 0.5)
        self.add(lecture_vgroup)

        # Animation Grid for Right Side (Mapping logical keys to screen coordinates)
        # 6x6 Grid on the right half of the screen
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]
        cols = ["1", "2", "3", "4", "5", "6"]
        
        start_x = 1.0
        start_y = 2.0
        step = 1.0

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Mapping positions to screen space
                x_pos = start_x + (j * step)
                y_pos = start_y - (i * step)
                self.grid[f"{row}{col}"] = np.array([x_pos, y_pos, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        """
        Utility to place mobjects on the right-side grid.
        """
        mobject.scale(scale_factor)
        if grid_pos in self.grid:
            mobject.move_to(self.grid[grid_pos])
        return mobject