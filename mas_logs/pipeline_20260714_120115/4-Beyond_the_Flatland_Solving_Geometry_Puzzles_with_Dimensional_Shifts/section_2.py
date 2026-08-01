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
        # Define lecture lines for Section 2 (from storyboard)
        lecture_lines = [
            "Let's build up from zero dimensions.",
            "A point extends into a one-dimensional line.",
            "The line expands into a two-dimensional square.",
            "Adding depth creates a three-dimensional cube.",
            "We can unfold 3D objects into 2D nets."
        ]
        
        self.setup_layout("Prerequisite Knowledge: The N+1 Foundation", lecture_lines)
        
        # Define brand colors
        YELLOW_C = "#FFFF00"
        RED_C = "#FF0000"
        
        # === Animation for Lecture Line 1 ===
        # Let's build up from zero dimensions.
        self.play(self.lecture[0].animate.set_color(YELLOW_C))
        dot = Dot(color=YELLOW_C)
        self.place_at_grid(dot, "B4", scale_factor=0.8)
        self.play(Create(dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A point extends into a one-dimensional line.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW_C)
        )
        # 1D line representation
        line_1d = Line(LEFT, RIGHT, color=YELLOW_C)
        self.place_at_grid(line_1d, "B4", scale_factor=1.0)
        self.play(ReplacementTransform(dot, line_1d))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The line expands into a two-dimensional square.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW_C)
        )
        # 2D square (start_square from Issue 61)
        start_square = Square(side_length=1.0, color=YELLOW_C, fill_opacity=0.3)
        self.place_at_grid(start_square, 'B4', scale_factor=0.8)
        self.play(ReplacementTransform(line_1d, start_square))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Adding depth creates a three-dimensional cube.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW_C)
        )
        # Pseudo-3D cube constructed with faces for transformation logic
        cube_faces = VGroup(*[Square(side_length=1.0, color=YELLOW_C, fill_opacity=0.3, stroke_width=2) for _ in range(6)])
        cube_faces[0].shift(0.25*UR) # Back face
        # Front face and others initially at origin
        for i in range(2, 6):
            cube_faces[i].move_to(cube_faces[1])
        
        cube_edges = VGroup()
        for i in range(4):
            cube_edges.add(Line(cube_faces[1].get_vertices()[i], cube_faces[0].get_vertices()[i], color=YELLOW_C))
        
        cube_3d = VGroup(cube_faces, cube_edges)
        self.place_at_grid(cube_3d, "B4", scale_factor=0.8)
        
        self.play(ReplacementTransform(start_square, cube_3d))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # We can unfold 3D objects into 2D nets.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW_C)
        )
        
        # Load the net asset [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/net.svg]
        cube_net = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/net.svg")
        cube_net.set_color(YELLOW_C)
        # Position net as per Issue 61
        self.place_in_area(cube_net, 'A3', 'E5', scale_factor=0.7)
        
        self.play(ReplacementTransform(cube_3d, cube_net))
        
        # Red path across the net as per Issue 61
        dist_line = Line(self.grid["A4"], self.grid["E4"], color=RED_C, stroke_width=4)
        self.place_in_area(dist_line, 'A4', 'E4', scale_factor=1.0)
        
        self.play(Create(dist_line))
        self.wait(2)
        
        # Cleanup colors
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
