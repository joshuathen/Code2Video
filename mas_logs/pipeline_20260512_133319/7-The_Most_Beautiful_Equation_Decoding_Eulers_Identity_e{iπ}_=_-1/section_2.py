from manim import *
import numpy as np

# Fix: Manim's path resolution fails if the file path contains curly braces (like {iπ}).
# Overwriting config.input_file with a clean string prevents the KeyError during initialization.
config.input_file = "section_2.py"

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
        # Section title and lecture lines
        title_str = "Prerequisite: The Nature of 'i' as a Steering Wheel"
        lines_str = [
            'Forget "imaginary"; think of i as a rotation operator.',
            'Multiplying by i turns any vector ninety degrees left.',
            "It's a steering wheel for the complex plane."
        ]
        self.setup_layout(title_str, lines_str)
        
        # Color definitions for visual matching
        COLOR_AXES = "#FFFFFF"
        COLOR_ROTATION = "#00FFFF" # Cyan
        COLOR_HIGHLIGHT = "#FFFF00" # Yellow
        COLOR_OPERATOR = "#00FF00" # Lime

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))
        
        # Construct the complex plane
        axes_length = 1.5
        # Create components relative to origin (0,0) first
        real_axis = Arrow(start=LEFT * axes_length, end=RIGHT * axes_length, color=COLOR_AXES, buff=0, stroke_width=2)
        imag_axis = Arrow(start=DOWN * axes_length, end=UP * axes_length, color=COLOR_AXES, buff=0, stroke_width=2)
        label_re = Text("Real", font_size=16, color=COLOR_AXES).next_to(real_axis, RIGHT, buff=0.1)
        label_im = Text("Imaginary", font_size=16, color=COLOR_AXES).next_to(imag_axis, UP, buff=0.1)
        unit_tick = Line(RIGHT * 1.0 + UP * 0.1, RIGHT * 1.0 + DOWN * 0.1, color=COLOR_AXES)
        unit_label = Text("1", font_size=16, color=COLOR_AXES).next_to(unit_tick, DOWN, buff=0.1)
        
        plane_group = VGroup(real_axis, imag_axis, label_re, label_im, unit_tick, unit_label)
        # Use grid logic to center the plane
        self.place_in_area(plane_group, "B2", "E5")
        
        # Identify coordinate system origin from placed objects
        plane_origin = real_axis.get_center()
        
        self.play(FadeIn(plane_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_ROTATION)
        )
        
        # Integrated Asset: steering wheel
        wheel = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/wheel.svg")
        wheel.set_color(COLOR_OPERATOR)
        # Centering wheel at origin using grid area
        self.place_in_area(wheel, "C3", "D4", scale_factor=0.4)
        
        # Create a vector pointing to '1' (at origin + RIGHT*1.0)
        vec = Arrow(start=plane_origin, end=plane_origin + RIGHT * 1.0, color=COLOR_ROTATION, buff=0, stroke_width=5)
        
        self.play(GrowArrow(vec), FadeIn(wheel))
        self.wait(0.5)
        
        # Rotate point from (1, 0) to (0, 1) (90 degrees counter-clockwise) using wheel asset
        self.play(
            Rotate(vec, angle=PI/2, about_point=plane_origin),
            Rotate(wheel, angle=PI/2, about_point=plane_origin),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_OPERATOR)
        )
        
        # Highlight 'i' at (0, 1) relative to origin
        i_pos = plane_origin + UP * 1.0
        i_dot = Dot(i_pos, color=COLOR_ROTATION)
        i_text = Text("i", font_size=28, color=COLOR_ROTATION).next_to(i_dot, UR, buff=0.1)
        
        # Rotation label at C4 (Issue 31 fix)
        rotation_label = Text("Rotation", font_size=22, color=COLOR_ROTATION)
        self.place_at_grid(rotation_label, 'C4', scale_factor=0.8)
        
        # Geometric Operator label in area E2-F5 (Issue 32 & 33 fix)
        steering_wheel_label = Text("Geometric Operator", font_size=18, color=COLOR_OPERATOR)
        self.place_in_area(steering_wheel_label, 'E2', 'F5', scale_factor=0.8)
        
        self.play(
            FadeIn(i_dot),
            Write(i_text),
            FadeIn(rotation_label),
            FadeIn(steering_wheel_label)
        )
        self.wait(2)
