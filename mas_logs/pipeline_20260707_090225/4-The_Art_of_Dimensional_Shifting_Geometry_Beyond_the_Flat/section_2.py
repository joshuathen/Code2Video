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
        # Initialize Layout
        lecture_lines = [
            'A zero-dimensional point marks a single location.', 
            'Stretching it creates lines, then flat 2D shapes.', 
            'Shifting into 3D transforms the square into a cube.'
        ]
        self.setup_layout("Prerequisite Knowledge: The 0D to 3D Ladder", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # A white point [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/point.svg] (#FFFFFF) appears in the center, labeled '0D: Point'.
        self.lecture[0].set_color(WHITE)
        point_asset = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/point.svg", color=WHITE)
        self.place_in_area(point_asset, "B3", "D5", scale_factor=0.4)
        
        point_label = Text("0D: Point", font_size=20, color=WHITE)
        self.place_at_grid(point_label, 'D4', scale_factor=0.8) # Issue 31 Fix
        
        self.play(FadeIn(point_asset), Write(point_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # The point stretches horizontally into a cyan line (#00FFFF) '1D', then vertically into a yellow square (#FFFF00) '2D'.
        
        # 1D transition
        self.lecture[1].set_color("#00FFFF")
        cyan_line = Line(LEFT, RIGHT, color="#00FFFF")
        self.place_in_area(cyan_line, "B3", "D5", scale_factor=2.0)
        label_1d = Text("1D", font_size=20, color="#00FFFF")
        label_1d.next_to(cyan_line, DOWN, buff=0.3)

        self.play(
            ReplacementTransform(point_asset, cyan_line),
            ReplacementTransform(point_label, label_1d)
        )
        self.wait(1.5)
        
        # 2D transition (Square)
        self.lecture[1].set_color("#FFFF00")
        square_2d = Square(side_length=2.2, color="#FFFF00", fill_opacity=0.3)
        self.place_in_area(square_2d, 'B3', 'D5', scale_factor=0.9) # Issue 32 Fix
        label_2d = Text("2D", font_size=20, color="#FFFF00")
        label_2d.next_to(square_2d, DOWN, buff=0.3)
        
        self.play(
            ReplacementTransform(cyan_line, square_2d),
            ReplacementTransform(label_1d, label_2d)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # The square stretches diagonally into an isometric magenta cube (#FF00FF) labeled '3D: Space'.
        self.lecture[2].set_color("#FF00FF")
        
        # Create Isometric Cube manually
        cube_color = "#FF00FF"
        size = 2.0
        offset = np.array([0.6, 0.6, 0])
        
        front_face = Square(side_length=size, color=cube_color, fill_opacity=0.1)
        back_face = Square(side_length=size, color=cube_color, fill_opacity=0.1).shift(offset)
        
        v_f = front_face.get_vertices()
        v_b = back_face.get_vertices()
        edges = VGroup(*[Line(v_f[i], v_b[i], color=cube_color) for i in range(4)])
        
        cube_group = VGroup(back_face, edges, front_face)
        # Issue 33 Fix: place_in_area 'B3' to 'E5'
        self.place_in_area(cube_group, 'B3', 'E5', scale_factor=0.8)
        
        label_3d = Text("3D: Space", font_size=20, color="#FF00FF")
        label_3d.next_to(cube_group, DOWN, buff=0.3)
        
        self.play(
            ReplacementTransform(square_2d, cube_group),
            ReplacementTransform(label_2d, label_3d)
        )
        self.wait(3)
