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
        # Setup layout with title and lecture lines
        self.setup_layout(
            "Leaping Higher: The Tesseract Challenge", 
            [
                "A moving point traces a one-dimensional line.", 
                "Sliding this line creates a two-dimensional square.", 
                "Shifting the square forms a three-dimensional cube.", 
                "A cube moving into the fourth dimension forms a Tesseract.", 
                "Rotating a Tesseract reveals its complex shifting structure."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        point = Dot(self.grid["C4"], color=WHITE)
        self.play(Create(point))
        
        line_1d = Line(self.grid["C4"], self.grid["C5"], color=WHITE)
        self.play(ReplacementTransform(point, line_1d))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        square_2d = Square(side_length=1.5, color="#00FFFF")
        # Fix Issue 37: self.place_in_area(square_2d, 'B3', 'D5', scale_factor=1.0)
        self.place_in_area(square_2d, 'B3', 'D5', scale_factor=1.0)
        
        self.play(
            ReplacementTransform(line_1d, square_2d)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        # We'll build a wireframe projection for the cube
        front_sq = Square(side_length=1.5, color=WHITE)
        back_sq = Square(side_length=1.5, color=WHITE).shift(0.6 * UR)
        connectors = VGroup(*[
            Line(front_sq.get_corner(c), back_sq.get_corner(c), color=WHITE)
            for c in [UL, UR, DL, DR]
        ])
        cube_3d = VGroup(front_sq, back_sq, connectors)
        # Fix Issue 38: self.place_in_area(cube_3d, 'B3', 'D5', scale_factor=1.0)
        self.place_in_area(cube_3d, 'B3', 'D5', scale_factor=1.0)

        self.play(
            square_2d.animate.set_color(WHITE),
            ReplacementTransform(square_2d, cube_3d)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF00FF")
        
        # Outer cube (larger)
        outer_front = Square(side_length=2.5, color="#FF00FF")
        outer_back = Square(side_length=2.5, color="#FF00FF").shift(0.7 * UR)
        outer_connectors = VGroup(*[
            Line(outer_front.get_corner(c), outer_back.get_corner(c), color="#FF00FF")
            for c in [UL, UR, DL, DR]
        ])
        outer_cube = VGroup(outer_front, outer_back, outer_connectors)
        
        # Inner cube (smaller)
        inner_front = Square(side_length=1.0, color="#FF00FF")
        inner_back = Square(side_length=1.0, color="#FF00FF").shift(0.3 * UR)
        inner_connectors = VGroup(*[
            Line(inner_front.get_corner(c), inner_back.get_corner(c), color="#FF00FF")
            for c in [UL, UR, DL, DR]
        ])
        inner_cube = VGroup(inner_front, inner_back, inner_connectors)
        
        # Tesseract base
        tesseract_base = VGroup(outer_cube, inner_cube)
        
        # Bridge lines between outer and inner cubes
        bridges = VGroup(*[
            Line(outer_front.get_corner(c), inner_front.get_corner(c), color="#FF00FF")
            for c in [UL, UR, DL, DR]
        ], *[
            Line(outer_back.get_corner(c), inner_back.get_corner(c), color="#FF00FF")
            for c in [UL, UR, DL, DR]
        ])
        
        full_tesseract = VGroup(tesseract_base, bridges)
        # Fix Issue 39: self.place_in_area(tesseract, 'B3', 'F6', scale_factor=0.9)
        self.place_in_area(full_tesseract, 'B3', 'F6', scale_factor=0.9)

        self.play(
            ReplacementTransform(cube_3d, full_tesseract),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF00FF")
        
        # Simulating 4D rotation by shifting vertices (scaling inner and outer cubes inversely)
        self.play(
            outer_cube.animate.scale(0.5).move_to(full_tesseract.get_center()),
            inner_cube.animate.scale(2.2).move_to(full_tesseract.get_center()),
            UpdateFromAlphaFunc(bridges, lambda m, a: m.become(VGroup(*[
                Line(outer_front.get_corner(c), inner_front.get_corner(c), color="#FF00FF")
                for c in [UL, UR, DL, DR]
            ], *[
                Line(outer_back.get_corner(c), inner_back.get_corner(c), color="#FF00FF")
                for c in [UL, UR, DL, DR]
            ]))),
            run_time=3,
            rate_func=there_and_back
        )
        
        self.wait(2)
