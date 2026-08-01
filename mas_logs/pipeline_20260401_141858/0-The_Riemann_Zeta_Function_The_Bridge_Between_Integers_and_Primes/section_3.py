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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The Basel Milestone: When Math Gets Beautiful", 
            [
                "Euler discovered zeta of two is pi squared over six.", 
                "We can unroll a circle to match this sum.", 
                "This reveals a bridge between integers and geometry."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Color lecture line to match formula result
        self.lecture[0].set_color("#FFD700")
        
        # Display the Basel series
        basel_series = VGroup(
            Text("1 + 1/4 + 1/9 + ... =", font_size=36),
            Text("π²/6", font_size=36)
        ).arrange(RIGHT, buff=0.2)
        
        # Set result color
        basel_series[1].set_color("#FFD700")
        self.place_in_area(basel_series, "A2", "B5", scale_factor=0.9)
        
        self.play(Write(basel_series))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color lecture line to match geometry
        self.lecture[1].set_color("#00FF00")
        
        # Draw a circle of radius \sqrt{\pi^2/6} / \pi in #00FF00
        radius_val = np.sqrt(np.pi**2 / 6) / np.pi
        circle = Circle(radius=radius_val, color="#00FF00")
        # [Issue 39] Move circle to Row C (C3)
        self.place_at_grid(circle, "C3", scale_factor=1.0)
        
        circle_label = Text("R = √(π²/6) / π", font_size=24, color="#00FF00")
        # [Issue 38] Position label in area C4-C6 to avoid overlap
        self.place_in_area(circle_label, "C4", "C6", scale_factor=0.8)
        
        self.play(Create(circle), Write(circle_label))
        self.wait(0.5)
        
        # Pulse the formula and the circle simultaneously in #FFD700
        self.play(
            basel_series.animate.set_color("#FFD700"),
            circle.animate.set_color("#FFD700"),
            run_time=0.5
        )
        self.play(
            basel_series[0].animate.set_color(WHITE),
            circle.animate.set_color("#00FF00"),
            run_time=0.5
        )
        self.wait(1)

        # Cut and unroll animation
        target_length = np.pi**2 / 6
        unrolled_line = Line(
            LEFT * (target_length / 2), 
            RIGHT * (target_length / 2), 
            color="#00FF00"
        )
        # [Issue 40] Position line at D3
        self.place_at_grid(unrolled_line, "D3", scale_factor=1.0)
        
        line_label = Text("L = π²/6", font_size=24, color="#FFD700")
        # Place label at D4
        self.place_at_grid(line_label, "D4", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(circle, unrolled_line),
            FadeOut(circle_label),
            run_time=2
        )
        self.play(Write(line_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color lecture line to match highlight
        self.lecture[2].set_color("#87CEEB")
        
        # Show connection with highlight box
        highlight_box = SurroundingRectangle(VGroup(basel_series[1], line_label), color="#87CEEB", buff=0.2)
        self.play(Create(highlight_box))
        self.wait(3)
