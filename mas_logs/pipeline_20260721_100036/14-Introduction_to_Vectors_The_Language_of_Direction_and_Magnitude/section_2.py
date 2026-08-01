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
        # Setup layout with title and lecture lines
        self.setup_layout("Prerequisite: The Coordinate Grid", [
            "The coordinate grid is our mathematical playground.",
            "We locate points using horizontal and vertical coordinates.",
            "Every position is defined by an (x, y) pair."
        ])

        # === Animation for Lecture Line 1 ===
        # Color change for current line
        self.lecture[0].set_color(WHITE)
        
        # Create a coordinate grid
        # We define a plane that covers roughly the grid area
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=5,
            y_length=5,
            background_line_style={
                "stroke_color": "#808080",
                "stroke_width": 2,
                "stroke_opacity": 0.6
            },
            axis_config={"stroke_color": "#FFFFFF", "include_tip": True}
        )
        # Fix for Issue 24: Apply scale_factor 0.85 and position in A1-F6
        self.place_in_area(plane, 'A1', 'F6', scale_factor=0.85)
        
        self.play(Create(plane), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We locate points using horizontal and vertical coordinates."
        # Color change for current line
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # Point at (3, 2)
        # Fix for Issue 25: Use place_at_grid for dot at 'B6' with scale_factor 1.5
        point_dot = Dot(color="#00FFFF")
        self.place_at_grid(point_dot, 'B6', scale_factor=1.5)
        
        # Fix for Issue 26: Use place_at_grid for coord_label at 'A6' with scale_factor 0.8
        coord_label = Text("(3, 2)", font_size=24, color="#00FFFF")
        self.place_at_grid(coord_label, 'A6', scale_factor=0.8)
        
        self.play(FadeIn(point_dot), Write(coord_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Every position is defined by an (x, y) pair."
        # Color change for current line
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        # Dashed lines to show measurements from origin to the point
        # Origin of plane is its center
        plane_origin = plane.get_center()
        target_pos = point_dot.get_center()
        
        # Horizontal path
        h_line = DashedLine(
            start=plane_origin,
            end=[target_pos[0], plane_origin[1], 0],
            color="#00FFFF",
            dash_length=0.1
        )
        # Vertical path
        v_line = DashedLine(
            start=[target_pos[0], plane_origin[1], 0],
            end=target_pos,
            color="#00FFFF",
            dash_length=0.1
        )
        
        self.play(Create(h_line), run_time=1)
        self.play(Create(v_line), run_time=1)
        self.wait(2)
