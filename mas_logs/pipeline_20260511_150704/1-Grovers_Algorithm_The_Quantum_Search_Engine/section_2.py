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
        # Setup context
        title_text = "Prerequisite: Quantum Superposition"
        lecture_lines = [
            'Quantum computers use superposition to check items simultaneously.',
            'We represent every possibility with equal probability amplitudes.',
            'All states start in a uniform flat sea.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # Define animation objects
        # 1. 100 Squares (Representing 100 physical boxes)
        squares = VGroup(*[
            Square(side_length=0.25, stroke_width=1, color=GREY_B) 
            for _ in range(100)
        ])
        squares.arrange_in_grid(rows=10, cols=10, buff=0.08)
        # Issue 35 fix: Updated placement area and scale
        self.place_in_area(squares, 'B1', 'D6', scale_factor=0.9)

        # 2. 100 Bars (Representing amplitudes)
        bars = VGroup(*[
            Rectangle(
                height=1.5, 
                width=0.03, 
                stroke_width=0, 
                fill_opacity=1, 
                fill_color="#5271FF"
            ) for _ in range(100)
        ])
        bars.arrange(RIGHT, buff=0.02, aligned_edge=DOWN)
        # Issue 36 fix: Updated scale factor
        self.place_in_area(bars, 'B1', 'D6', scale_factor=0.9)
        
        # 3. Dashed line (Flat sea)
        # Position at the top of the bars
        top_y = bars.get_top()[1]
        dashed_line = DashedLine(
            start=[bars.get_left()[0], top_y, 0],
            end=[bars.get_right()[0], top_y, 0],
            color="#00FFFF",
            stroke_width=2
        )

        # 4. Label
        label = Text(
            "Quantum Superposition: All states exist simultaneously", 
            font_size=18, 
            color=WHITE
        )
        # Issue 37 fix: Added scale_factor for better fit
        self.place_in_area(label, 'F1', 'F6', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Intro: Show physical boxes, then morph to quantum superposition bars
        self.add(squares)
        self.wait(1)
        self.play(
            self.lecture[0].animate.set_color("#5271FF"),
            ReplacementTransform(squares, bars),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Explain amplitude concept
        self.play(
            self.lecture[1].animate.set_color("#5271FF"),
            bars.animate.set_fill(opacity=0.6),
            run_time=0.8
        )
        self.play(
            bars.animate.set_fill(opacity=1),
            run_time=0.8
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Draw the uniform "sea" dashed line and display text label
        self.play(
            self.lecture[2].animate.set_color("#00FFFF"),
            Create(dashed_line),
            Write(label),
            run_time=2
        )
        self.wait(3)
