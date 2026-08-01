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
        # Initial layout setup
        self.setup_layout(
            "The Time Equation Construction", 
            [
                "Total time is distance divided by speed in each medium.", 
                "We use Pythagoras to find lengths L1 and L2.", 
                "Time T(x) equals L1/v1 plus L2/v2."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Using Text instead of MathTex to avoid FileNotFoundError: 'latex'
        time_speed_formula = Text(
            "Time = Distance / Speed", 
            color="#00FF00",
            font_size=32
        )
        self.place_in_area(time_speed_formula, 'A1', 'A6', scale_factor=1.0)
        
        self.play(
            Write(time_speed_formula),
            self.lecture[0].animate.set_color("#00FF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Using Text with unicode characters for math representations
        l1_eq = Text("L₁ = sqrt(x² + a²)", color="#FFFF00", font_size=28)
        l2_eq = Text("L₂ = sqrt((d-x)² + b²)", color="#FF00FF", font_size=28)
        
        self.place_at_grid(l1_eq, 'B2', scale_factor=0.8)
        self.place_at_grid(l2_eq, 'D5', scale_factor=0.8)
        
        self.play(
            Write(l1_eq),
            Write(l2_eq),
            self.lecture[1].animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Text-based substitution for the total time equation
        total_time_eq = Text(
            "T(x) = sqrt(x² + a²) / v₁ + sqrt((d-x)² + b²) / v₂", 
            color="#00FF00",
            font_size=24
        )
        self.place_in_area(total_time_eq, 'F1', 'F6', scale_factor=0.9)
        
        self.play(
            Write(total_time_eq),
            self.lecture[2].animate.set_color("#00FF00")
        )
        self.wait(2)
