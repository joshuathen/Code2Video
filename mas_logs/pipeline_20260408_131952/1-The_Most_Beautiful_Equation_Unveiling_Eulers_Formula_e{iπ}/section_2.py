from manim import *
import numpy as np
from pathlib import Path

# Fix: Manim CE v0.19.0 configuration system bug with curly braces in file paths.
# The config.get_dir method uses a 'while "{" in path' loop which attempts to 
# format the path as a string template, causing a KeyError when it finds characters like {iπ}.
import manim
_input_path_str = str(manim.config.input_file)
if "{" in _input_path_str or "}" in _input_path_str:
    # Replacing braces with underscores prevents the recursive formatting loop.
    manim.config.input_file = Path(_input_path_str.replace("{", "_").replace("}", "_"))

# === TeachingScene Base Class ===
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

# === Section 2 Scene ===
class Section2Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        title_text = "Prerequisite: The Complex Plane"
        lecture_lines = [
            "Visualize numbers on the complex number plane.",
            "The vertical axis represents the imaginary unit i.",
            "Multiplying by i rotates a point ninety degrees."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Description: A horizontal Real axis and a vertical Imaginary axis appear in #FFFFFF.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        origin_pos = self.grid['D3']
        real_axis = Arrow(start=self.grid['D1'], end=self.grid['D6'], color=WHITE, buff=0, tip_length=0.2)
        imag_axis = Arrow(start=self.grid['F3'], end=self.grid['A3'], color=WHITE, buff=0, tip_length=0.2)
        
        # Issue 23 Fix: Relocate re_label to avoid overlap with arrowhead
        re_label = Text("Re", font_size=18, color=WHITE)
        self.place_at_grid(re_label, 'E6', scale_factor=0.8)
        
        # Issue 24 Fix: Relocate im_label to avoid overlap with arrowhead
        im_label = Text("Im", font_size=18, color=WHITE)
        self.place_at_grid(im_label, 'A4', scale_factor=0.8)
        
        self.play(Create(real_axis), Create(imag_axis))
        self.play(Write(re_label), Write(im_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Description: A point appears at (1, 0) on the Real axis, labeled '1' in #00FF00.
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        point_1 = Dot(point=self.grid['D4'], color="#00FF00")
        label_1 = Text("1", color="#00FF00", font_size=32)
        self.place_at_grid(label_1, 'E4', scale_factor=1.0)
        
        self.play(FadeIn(point_1), Write(label_1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Description: The point rotates 90 degrees counter-clockwise to (0, 1), labeled 'i' in #FF0000.
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        
        # Create a rotation arc path (radius=1.0)
        rot_path = Arc(
            radius=1.0, 
            start_angle=0, 
            angle=PI/2, 
            arc_center=origin_pos, 
            color=WHITE, 
            stroke_width=1
        ).set_opacity(0.3)
        
        # Target label 'i' at C3
        label_i = Text("i", color="#FF0000", font_size=32, slant=ITALIC)
        # Position 'i' near its final location (C2 or B3 might work better than exact C3 center)
        # Using B3 to be above the final point at C3
        dummy_label_i = Text("i", color="#FF0000", font_size=32, slant=ITALIC)
        self.place_at_grid(dummy_label_i, 'C2', scale_factor=1.0)
        
        self.play(Create(rot_path), run_time=0.5)
        self.play(
            MoveAlongPath(point_1, rot_path),
            Transform(label_1, dummy_label_i),
            point_1.animate.set_color("#FF0000"),
            run_time=2
        )
        self.play(FadeOut(rot_path))
        self.wait(2)
