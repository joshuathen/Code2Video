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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "The Classical Search Dilemma"
        lines = [
            'Searching unstructured databases is a common challenge.', 
            'For N items, we check each one sequentially.', 
            'Classically, finding a target takes O(N) steps.'
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Create a 10x10 grid of 100 white squares
        self.lecture[0].set_color(WHITE)
        
        boxes = VGroup(*[
            Square(side_length=0.3, color="#FFFFFF", stroke_width=1.5) 
            for _ in range(100)
        ]).arrange_in_grid(rows=10, cols=10, buff=0.05)
        
        # Place boxes in the center-right area (A1 to E6)
        # Issue 34 Fix: Scale factor reduced to 0.9 for breathing room
        self.place_in_area(boxes, "A1", "E6", scale_factor=0.9)
        
        self.play(Create(boxes), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight line 2 in yellow
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Create a yellow border to highlight boxes
        highlight_border = Square(side_length=0.32, color="#FFFF00", stroke_width=3)
        highlight_border.move_to(boxes[0])
        self.add(highlight_border)

        # Move the border sequentially through the first 50 boxes
        # Using a loop for the sequence
        for i in range(1, 51):
            # Speed up the movement gradually
            duration = max(0.02, 0.2 - (i * 0.005))
            self.play(highlight_border.animate.move_to(boxes[i]), run_time=duration, rate_func=linear)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight line 3 in white (matching text)
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        # Display text 'Classical Search: O(N) complexity' at the bottom (F1-F6 area)
        complexity_text = Text("Classical Search: O(N) complexity", font_size=24, color="#FFFFFF")
        # Issue 33 Fix: Scale factor reduced to 0.7 to avoid clipping
        self.place_in_area(complexity_text, "F1", "F6", scale_factor=0.7)
        
        self.play(Write(complexity_text))
        self.wait(3)
