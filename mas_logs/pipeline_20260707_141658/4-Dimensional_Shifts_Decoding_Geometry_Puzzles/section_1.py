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

class Section1Scene(TeachingScene):
    def construct(self):
        title_str = "Foundation: The Ladder of Dimensions"
        lecture_lines = [
            "A zero-dimensional point has no size or direction.",
            "Dragging a point creates a one-dimensional line.",
            "Shifting a line sideways produces a two-dimensional plane.",
            "Moving a plane upward builds a three-dimensional volume.",
            "Each shift adds a new direction to our space."
        ]
        self.setup_layout(title_str, lecture_lines)

        # Initialize all lecture lines to gray
        for line in self.lecture:
            line.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # "A zero-dimensional point has no size or direction."
        self.play(self.lecture[0].animate.set_color(WHITE))
        dot = Dot(color="#FFFFFF")
        # Fix: Line 66: self.place_at_grid(dot, 'D3', scale_factor=0.8)
        self.place_at_grid(dot, 'D3', scale_factor=0.8)
        self.play(FadeIn(dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Dragging a point creates a one-dimensional line."
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color("#FFD700")
        )
        
        line_start = self.grid["D3"]
        line_end = self.grid["D5"]
        line_obj = Line(line_start, line_end, color="#FFD700", stroke_width=6)
        
        self.play(
            dot.animate.move_to(line_end),
            Create(line_obj)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Shifting a line sideways produces a two-dimensional plane."
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color("#00BFFF")
        )
        
        # Plane area D3 to E4
        # Fix: Line 103: self.place_in_area(plane_rect, 'D3', 'E4', scale_factor=0.8)
        plane_rect = Rectangle(
            width=1.0, 
            height=1.0, 
            fill_color="#00BFFF",
            fill_opacity=0.5,
            stroke_color="#00BFFF"
        )
        self.place_in_area(plane_rect, 'D3', 'E4', scale_factor=0.8)
        
        self.play(
            FadeOut(dot),
            ReplacementTransform(line_obj, plane_rect)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Moving a plane upward builds a three-dimensional volume."
        self.play(
            self.lecture[2].animate.set_color(GRAY),
            self.lecture[3].animate.set_color("#FF4500")
        )
        
        # Use Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cube.svg
        # Place in area D3 to F5 to avoid Column B (Issue 23)
        cube_asset = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cube.svg")
        cube_asset.set_color("#FF4500")
        self.place_in_area(cube_asset, "D3", "F5", scale_factor=1.5)
        
        self.play(
            ReplacementTransform(plane_rect, cube_asset)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Each shift adds a new direction to our space."
        self.play(
            self.lecture[3].animate.set_color(GRAY),
            self.lecture[4].animate.set_color(WHITE)
        )
        
        # Origin for axes (center of the cube)
        origin = cube_asset.get_center()
        
        x_axis = Arrow(origin, origin + RIGHT * 1.5, color=RED, buff=0, stroke_width=4)
        y_axis = Arrow(origin, origin + UP * 1.5, color=GREEN, buff=0, stroke_width=4)
        z_axis = Arrow(origin, origin + np.array([-0.8, -0.8, 0]), color=BLUE, buff=0, stroke_width=4)
        
        x_label = Text("X", font_size=18, color=RED).next_to(x_axis, RIGHT, buff=0.1)
        y_label = Text("Y", font_size=18, color=GREEN).next_to(y_axis, UP, buff=0.1)
        z_label = Text("Z", font_size=18, color=BLUE).next_to(z_axis, DL, buff=0.1)
        
        axes_group = VGroup(x_axis, y_axis, z_axis, x_label, y_label, z_label)
        
        self.play(Create(axes_group))
        self.wait(3)
