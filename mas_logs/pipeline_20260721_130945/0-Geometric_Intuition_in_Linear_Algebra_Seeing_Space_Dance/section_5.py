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

class Section5Scene(TeachingScene):
    def construct(self):
        # Data from storyboard
        title_text = "The Determinant: Scaling the Area"
        lecture_lines = [
            "The determinant measures how space is scaled or squished.",
            "It's the area of the transformed unit square.",
            "A determinant of four means the area quadruples.",
            "A determinant of zero squishes the world into one dimension.",
            "Pixie's two-dimensional world loses all its area."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Define colors for elements
        c_square = "#FFFF00"
        c_label = "#FFFFFF"
        c_highlight = "#FFFF00"

        # Pre-calculate the center of the transformation area (C3 to D4)
        target_center = (self.grid['C3'] + self.grid['D4']) / 2

        # === Animation for Lecture Line 1 ===
        # "The determinant measures how space is scaled or squished."
        self.play(self.lecture[0].animate.set_color(c_highlight))
        
        # [Animation 1] Draw unit square (0,0) to (1,1) filled with #FFFF00 at 0.3 opacity.
        # Fix for Issue 37: Increase scale_factor to 1.4 for better grid utilization.
        unit_square = Square(side_length=1.0, color=c_square, stroke_width=2)
        unit_square.set_fill(c_square, opacity=0.3)
        self.place_in_area(unit_square, 'C3', 'D4', scale_factor=1.4)
        
        self.play(FadeIn(unit_square))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "It's the area of the transformed unit square."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(c_highlight)
        )
        
        # [Animation 2] Add label "Area = 1" in #FFFFFF inside the square.
        area_label = Text("Area = 1", font_size=18, color=c_label)
        area_label.move_to(target_center)
        
        self.play(Write(area_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "A determinant of four means the area quadruples."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(c_highlight)
        )
        
        # [Animation 3 & 4] Apply scaling matrix [[2, 0], [0, 2]] and change label to "Area = 4"
        # Fix for Issue 38: Position "Area = 4" label at 'D4' with scale_factor 0.9.
        new_area_label = Text("Area = 4", font_size=18, color=c_label)
        self.place_at_grid(new_area_label, 'D4', scale_factor=0.9)
        
        self.play(
            unit_square.animate.apply_matrix([[2, 0], [0, 2]], about_point=target_center),
            ReplacementTransform(area_label, new_area_label),
            run_time=2
        )
        area_label = new_area_label
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "A determinant of zero squishes the world into one dimension."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(c_highlight)
        )
        
        # [Animation 5] Apply matrix [[1, 1], [1, 1]] to collapse square to line; label "Area = 0".
        # Fix for Issue 36: Position "Area = 0" label at 'C5' with scale_factor 0.8 to avoid overlap with line.
        zero_area_label = Text("Area = 0", font_size=18, color=c_label)
        self.place_at_grid(zero_area_label, 'C5', scale_factor=0.8)
        
        self.play(
            unit_square.animate.apply_matrix([[1, 1], [1, 1]], about_point=target_center),
            ReplacementTransform(area_label, zero_area_label),
            run_time=2
        )
        area_label = zero_area_label
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Pixie's two-dimensional world loses all its area."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(c_highlight)
        )
        
        # Final emphasis using Indicate as per L004.
        self.play(Indicate(unit_square, color=c_square))
        self.wait(2)
        
        # Reset highlighting for completion
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
