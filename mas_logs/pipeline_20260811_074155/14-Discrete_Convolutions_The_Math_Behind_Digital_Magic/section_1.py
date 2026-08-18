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
        self.setup_layout("The Hook: How Robots 'See' Features", [
            "Meet Pixel, a robot who sees the world as data.",
            "To find objects, Pixel must identify edges and patterns.",
            "Convolution converts these raw grids into meaningful visual features."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Robot head asset (#00FFFF)
        pixel_color = "#00FFFF"
        robot_head = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg", color=pixel_color)
        # Use place_in_area as requested in Issue 33
        self.place_in_area(robot_head, 'B2', 'E3', scale_factor=0.8)
        
        self.play(
            self.lecture[0].animate.set_color(pixel_color),
            FadeIn(robot_head),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Simple cube (#FFFF00) with glowing lines
        cube_color = "#FFFF00"
        # Front face
        f1 = np.array([-0.5, -0.5, 0])
        f2 = np.array([ 0.5, -0.5, 0])
        f3 = np.array([ 0.5,  0.5, 0])
        f4 = np.array([-0.5,  0.5, 0])
        # Back face
        off = 0.4
        b1 = np.array([-0.5+off, -0.5+off, 0])
        b2 = np.array([ 0.5+off, -0.5+off, 0])
        b3 = np.array([ 0.5+off,  0.5+off, 0])
        b4 = np.array([-0.5+off,  0.5+off, 0])
        
        front_square = Polygon(f1, f2, f3, f4, color=cube_color)
        back_square = Polygon(b1, b2, b3, b4, color=cube_color)
        conn1 = Line(f1, b1, color=cube_color)
        conn2 = Line(f2, b2, color=cube_color)
        conn3 = Line(f3, b3, color=cube_color)
        conn4 = Line(f4, b4, color=cube_color)
        
        cube = VGroup(front_square, back_square, conn1, conn2, conn3, conn4)
        # Use scale_factor=0.7 as requested in Issue 33
        self.place_in_area(cube, 'B4', 'E6', scale_factor=0.7)
        
        # Highlighting edges with glowing lines
        glow_cube = cube.copy().set_stroke(width=8).set_color(cube_color).set_alpha(0.5)
        
        self.play(
            self.lecture[1].animate.set_color(cube_color),
            Create(cube),
            run_time=2
        )
        self.play(
            Create(glow_cube),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Transition edges into a simple matrix of numbers (#FFFFFF)
        matrix_color = "#FFFFFF"
        matrix_data = [
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ]
        
        matrix_mobs = VGroup()
        for r, row in enumerate(matrix_data):
            for c, val in enumerate(row):
                num = Text(str(val), font_size=24, color=matrix_color)
                num.shift(RIGHT * (c - 1) * 0.8 + DOWN * (r - 1) * 0.8)
                matrix_mobs.add(num)
        
        # Position matrix where the cube was, with scale_factor=0.8 as requested
        self.place_in_area(matrix_mobs, 'B4', 'E6', scale_factor=0.8)
        
        self.play(
            self.lecture[2].animate.set_color(matrix_color),
            FadeOut(cube),
            FadeOut(glow_cube),
            FadeIn(matrix_mobs),
            run_time=2
        )
        self.wait(2)
