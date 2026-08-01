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
        # Data for Section 4
        title = "The Determinant: The Scaling Factor"
        lecture_lines = [
            "The determinant measures how much areas scale during transformation.",
            "A determinant of two means the area has doubled.",
            "If it is zero, the space squishes into a line."
        ]
        self.setup_layout(title, lecture_lines)

        # Color palette
        color_stage1 = "#ADD8E6"  # Light blue
        color_stage2 = "#00008B"  # Dark blue
        color_stage3 = "#FF0000"  # Red
        text_white = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(color_stage1))

        # Create unit square
        # A 1x1 rectangle centered at some grid position
        square = Rectangle(width=1.0, height=1.0, fill_opacity=0.6, fill_color=color_stage1, stroke_color=color_stage1)
        # Refinement 1: Move square right to avoid crowding (C4-E6)
        self.place_in_area(square, 'C4', 'E6', scale_factor=0.6)
        
        # Ground origin for transformations
        origin = square.get_corner(DL)
        
        # Unit vectors
        i_hat = Arrow(start=origin, end=square.get_corner(DR), buff=0, color=WHITE, stroke_width=4)
        j_hat = Arrow(start=origin, end=square.get_corner(UL), buff=0, color=WHITE, stroke_width=4)
        
        # Labels
        # Refinement 2: Align area labels with new visual center (B5)
        area_label = Text("Area = 1", font_size=20, color=color_stage1)
        self.place_at_grid(area_label, 'B5', scale_factor=0.8)
        
        self.play(
            Create(square),
            Create(i_hat),
            Create(j_hat),
            Write(area_label)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line, reset first
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_stage2)
        )

        # Det indicator
        # Refinement 3: Align determinant labels with new center (A5)
        det_label = Text("Det = 2", font_size=24, color=text_white)
        self.place_at_grid(det_label, 'A5', scale_factor=0.8)
        
        # Updated area label
        new_area_label = Text("Area = 2", font_size=20, color=color_stage2)
        self.place_at_grid(new_area_label, 'B5', scale_factor=0.8)

        # Scaling horizontal area (Det = 2)
        # We scale the square and i-hat by 2 horizontally
        self.play(
            square.animate.scale(np.array([2, 1, 1]), about_point=origin).set_fill(color_stage2).set_stroke(color_stage2),
            i_hat.animate.scale(2, about_point=origin),
            Transform(area_label, new_area_label),
            Write(det_label),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line, reset second
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_stage3)
        )

        # Det indicator update
        zero_det_label = Text("Det = 0", font_size=24, color=text_white)
        self.place_at_grid(zero_det_label, 'A5', scale_factor=0.8)
        
        # Zero area label
        zero_area_label = Text("Area = 0", font_size=20, color=color_stage3)
        self.place_at_grid(zero_area_label, 'B5', scale_factor=0.8)

        # Squishing the y-axis to zero (Det = 0)
        self.play(
            square.animate.scale(np.array([1, 0.001, 1]), about_point=origin).set_fill(color_stage3).set_stroke(color_stage3),
            j_hat.animate.scale(0.001, about_point=origin),
            Transform(area_label, zero_area_label),
            Transform(det_label, zero_det_label),
            run_time=2
        )
        self.wait(3)
