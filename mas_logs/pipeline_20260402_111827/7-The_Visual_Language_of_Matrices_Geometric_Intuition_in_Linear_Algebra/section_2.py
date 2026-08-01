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
        # Initial Setup with correct lecture lines from snapshot
        title = "The Concept of Linear Transformation"
        lecture_lines = [
            "A linear transformation warps the space around us.",
            "The origin remains fixed throughout the transformation.",
            "Grid lines stay parallel and evenly spaced."
        ]
        self.setup_layout(title, lecture_lines)

        # Pre-define colors for readability
        LINE_HIGHLIGHT_COLOR = YELLOW
        GRID_COLOR = "#444444"
        ORIGIN_COLOR = "#FFFFFF"
        FLASH_COLOR = "#AAFFCC"

        # 1. Initialize the coordinate grid (NumberPlane)
        # Using fix from Issue 48: self.place_in_area(plane, 'A2', 'F6', scale_factor=0.9)
        plane = NumberPlane(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            background_line_style={
                "stroke_color": GRID_COLOR,
                "stroke_width": 2,
                "stroke_opacity": 0.6
            },
            axis_config={"stroke_color": GRID_COLOR}
        )
        self.place_in_area(plane, 'A2', 'F6', scale_factor=0.9)
        
        # 2. Origin Asset integration - Issue 42
        # Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/null.svg
        origin_dot = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/null.svg")
        origin_dot.set_color(ORIGIN_COLOR)
        # Placing in the same area as the plane ensures it's at (0,0)
        self.place_in_area(origin_dot, 'A2', 'F6', scale_factor=0.15)

        # 3. Origin Label - Issue 49 fix: self.place_at_grid(origin_label, 'D4', scale_factor=0.5)
        origin_label = Text("(0,0)", font_size=16, color=ORIGIN_COLOR)
        self.place_at_grid(origin_label, 'D4', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        # "A linear transformation warps the space around us."
        self.play(self.lecture[0].animate.set_color(LINE_HIGHLIGHT_COLOR))
        self.play(Create(plane), run_time=1)
        
        # Perform the shear transformation
        shear_matrix = [[1, 1], [0, 1]]
        self.play(
            plane.animate.apply_matrix(shear_matrix),
            run_time=2,
            rate_func=smooth
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The origin remains fixed throughout the transformation."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(LINE_HIGHLIGHT_COLOR)
        )
        # Show origin remains fixed (it is already at the center of the transformation)
        self.play(FadeIn(origin_dot), Write(origin_label))
        self.play(Indicate(origin_dot, color=ORIGIN_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Grid lines stay parallel and evenly spaced."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(LINE_HIGHLIGHT_COLOR)
        )
        
        # Flash the grid lines to demonstrate they are parallel and evenly spaced
        flash_lines = plane.copy().set_color(FLASH_COLOR).set_stroke(width=4)
        self.play(
            Indicate(flash_lines, color=FLASH_COLOR, scale_factor=1.02),
            run_time=2
        )
        self.remove(flash_lines)
        self.wait(2)

        # Final cleanup: Highlight reset
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
