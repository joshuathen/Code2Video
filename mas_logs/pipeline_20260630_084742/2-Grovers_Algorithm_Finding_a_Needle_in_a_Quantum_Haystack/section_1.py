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
        # Setup layout
        title_text = "The Problem: Searching in the Dark"
        lecture_lines = [
            "Imagine searching an unstructured database for one specific item.",
            "Classically, we check items one by one.",
            "For N items, we must check N/2 boxes."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create 10 white square outlines arranged in a horizontal row
        boxes = VGroup(*[Square(side_length=0.6, color=WHITE) for _ in range(10)])
        boxes.arrange(RIGHT, buff=0.2)
        
        # Place in the grid area C1 to C6
        # Fix for Issue 24: Adjusted grid area and scale to fit better vertically.
        self.place_in_area(boxes, "C1", "C6", scale_factor=0.65)
        
        self.play(Create(boxes))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line, reset first
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Move a yellow circle to each box; boxes change color to gray one-by-one.
        circle = Circle(radius=0.15, color=YELLOW, fill_opacity=1)
        # Start at first box
        circle.move_to(boxes[0].get_center())
        self.play(FadeIn(circle))
        
        # Animate checking 5 boxes (N/2 for N=10)
        for i in range(5):
            target_pos = boxes[i].get_center()
            if i > 0:
                self.play(circle.animate.move_to(target_pos), run_time=0.4)
            
            self.play(boxes[i].animate.set_color("#888888"), run_time=0.2)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line, reset second
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Display green text: "Classical search: N/2 attempts on average"
        summary_text = Text("Classical search: N/2 attempts on average", font_size=24, color="#00FF00")
        # Fix for Issue 23: Used place_in_area to center the text across the bottom area and prevent clipping.
        self.place_in_area(summary_text, "E1", "F6", scale_factor=0.7)
        
        self.play(Write(summary_text))
        self.wait(2)
