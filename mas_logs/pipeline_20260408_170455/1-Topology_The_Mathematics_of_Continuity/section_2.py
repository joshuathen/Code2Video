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
        # Setup layout with the title and the three lecture lines
        self.setup_layout(
            "Prerequisite: Continuous Deformation", 
            [
                'We follow strict rules called "continuous deformations."', 
                'You can stretch, bend, or shrink any shape freely.', 
                'However, tearing holes or gluing parts is strictly forbidden.'
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Line 1: 'We follow strict rules called "continuous deformations."'
        self.play(self.lecture[0].animate.set_color("#F0E442"))
        
        # Display a horizontal yellow line segment
        deformation_shape = Line(start=LEFT*2, end=RIGHT*2, color="#F0E442", stroke_width=6)
        # Fix Issue 34: Position the deformation shape correctly using place_in_area
        self.place_in_area(deformation_shape, 'C1', 'E6', scale_factor=0.8)
        
        self.play(Create(deformation_shape))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: 'You can stretch, bend, or shrink any shape freely.'
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#F0E442")
        )
        
        # Create a winding curve to represent the stretched/bent version
        curve_points = [
            LEFT*2,
            LEFT*1 + UP*1.0,
            ORIGIN + DOWN*1.0,
            RIGHT*1 + UP*1.0,
            RIGHT*2
        ]
        winding_curve = VMobject(color="#F0E442", stroke_width=6)
        winding_curve.set_points_smoothly(curve_points)
        # Position winding curve in the same area for consistent transformation
        self.place_in_area(winding_curve, 'C1', 'E6', scale_factor=0.8)
        
        self.play(Transform(deformation_shape, winding_curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: 'However, tearing holes or gluing parts is strictly forbidden.'
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#D55E00")
        )
        
        # Display a dashed line representing a 'cut'
        tear_line = DashedLine(
            start=UP * 0.4, 
            end=DOWN * 0.4, 
            color=WHITE, 
            stroke_width=4
        )
        # Fix Issue 35: Position the tear line using place_at_grid at 'C2'
        self.place_at_grid(tear_line, 'C2', scale_factor=1.0)
        
        # Red 'X' mark over the tear line
        x_mark = Cross(tear_line, stroke_color="#D55E00", stroke_width=10)
        
        self.play(Create(tear_line))
        self.play(Create(x_mark))
        self.wait(2)
