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
        # Fetching storyboard data
        title = "The Determinant: Area Scaling Factor"
        lines = [
            "The determinant measures how much areas are scaled.",
            "If the determinant is two, the area doubles.",
            "A determinant of zero squashes space into a line."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors
        light_green = "#90EE90"
        white = "#FFFFFF"
        yellow = "#FFFF00"
        red = "#FF0000"
        blue = "#0000FF"
        
        # Assets
        square_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/square.svg"
        
        # === Animation for Lecture Line 1 ===
        # Draw a unit square (1x1) using [Asset: ...] in light green (#90EE90) defined by basis vectors.
        # Label the internal area '1' with white text (#FFFFFF).
        
        self.lecture[0].set_color(light_green)
        
        # Load asset
        unit_square = SVGMobject(square_path).set_color(light_green).set_fill(light_green, opacity=0.3)
        # SVGMobject might not be 1x1 by default, let's set height/width
        unit_square.set_height(1)
        unit_square.set_width(1)
        
        area_label = Text("1", font_size=36, color=white)
        
        # Basis vectors for visual context
        i_hat = Arrow(start=ORIGIN, end=RIGHT, buff=0, color=blue, stroke_width=4)
        j_hat = Arrow(start=ORIGIN, end=UP, buff=0, color=RED, stroke_width=4)
        
        # Align square to origin (it's center-aligned by default)
        unit_square.move_to(RIGHT*0.5 + UP*0.5)
        area_label.move_to(unit_square.get_center())
        
        basis_group = VGroup(unit_square, i_hat, j_hat, area_label)
        # Issue 36: place_in_area(basis_group, 'B2', 'D5', scale_factor=1.0)
        self.place_in_area(basis_group, 'B2', 'D5', scale_factor=1.0)
        
        self.play(Create(unit_square), GrowArrow(i_hat), GrowArrow(j_hat))
        self.play(Write(area_label))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Transform the square into a larger parallelogram with area 6.
        # Fill the parallelogram with a semi-transparent yellow (#FFFF00).
        # Flash the text 'Determinant = 6' to highlight the area scaling factor.
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(yellow)
        
        # Target parallelogram (Matrix [[3, 1], [0, 2]] gives Det=6)
        # Use same relative coordinates as the unit square group
        # Parallelogram vertices relative to unit square's "origin"
        target_para = Polygon(
            ORIGIN, [3, 0, 0], [4, 2, 0], [1, 2, 0],
            stroke_color=yellow, fill_color=yellow, fill_opacity=0.5
        )
        target_i = Arrow(start=ORIGIN, end=[3, 0, 0], buff=0, color=blue, stroke_width=4)
        target_j = Arrow(start=ORIGIN, end=[1, 2, 0], buff=0, color=RED, stroke_width=4)
        
        # Move target shapes to where basis_group's "origin" is
        # Since we used place_in_area, let's calculate the offset
        origin_offset = basis_group.get_center() - (RIGHT*0.5 + UP*0.5)
        target_para.shift(origin_offset)
        target_i.shift(origin_offset)
        target_j.shift(origin_offset)
        
        area_6_label = Text("Area = 6", font_size=36, color=white)
        area_6_label.move_to(target_para.get_center())
        
        det_text = Text("Determinant = 6", font_size=32, color=yellow)
        # Issue 35: place_in_area(det_text, 'E2', 'E5', scale_factor=0.9)
        self.place_in_area(det_text, 'E2', 'E5', scale_factor=0.9)
        
        self.play(
            Transform(unit_square, target_para),
            Transform(i_hat, target_i),
            Transform(j_hat, target_j),
            Transform(area_label, area_6_label)
        )
        self.play(Flash(det_text, color=yellow), Write(det_text))
        self.wait(2)
        
        # === Animation for Lecture Line 3 ===
        # A determinant of zero squashes space into a line.
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(red)
        
        # Det 0: Matrix [[1.5, 1.5], [1.5, 1.5]] -> Area 0
        zero_line = Line(start=ORIGIN, end=[3, 3, 0], color=red, stroke_width=6)
        zero_i = Arrow(start=ORIGIN, end=[1.5, 1.5, 0], buff=0, color=blue, stroke_width=4)
        zero_j = Arrow(start=ORIGIN, end=[1.5, 1.5, 0], buff=0, color=RED, stroke_width=4)
        
        zero_line.shift(origin_offset)
        zero_i.shift(origin_offset)
        zero_j.shift(origin_offset)
        
        det_zero_text = Text("Determinant = 0", font_size=32, color=red)
        # Issue 37: place_in_area(det_zero_text, 'E2', 'E5', scale_factor=0.9)
        self.place_in_area(det_zero_text, 'E2', 'E5', scale_factor=0.9)
        
        self.play(
            Transform(unit_square, zero_line),
            Transform(i_hat, zero_i),
            Transform(j_hat, zero_j),
            FadeOut(area_label),
            Transform(det_text, det_zero_text)
        )
        self.wait(3)
