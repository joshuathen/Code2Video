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
        self.setup_layout(
            "The Chaos of the Real World", 
            [
                "Most data in our world looks chaotic and messy.",
                "Height distributions or dice rolls often seem completely random.",
                "Is there a hidden order within this universal randomness?"
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # "Most data in our world looks chaotic and messy."
        # A jagged, skewed bar chart representing random data appears in yellow (#FFFF00).
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        bar_heights = [0.8, 2.1, 0.4, 1.8, 3.2, 0.7, 1.5, 2.5]
        bars = VGroup(*[
            Rectangle(width=0.4, height=h, fill_color=YELLOW, fill_opacity=0.7, stroke_color=YELLOW)
            for h in bar_heights
        ]).arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        
        # Fixed Issue 22: Changed from B1, E6 to C1, F6 to reduce crowding near title.
        self.place_in_area(bars, "C1", "F6", scale_factor=0.8)
        self.play(Create(bars))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Height distributions or dice rolls often seem completely random."
        # Individual bars in the chart flash red (#FF0000) to show inconsistency.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(RED)
        )
        
        flash_indices = [0, 2, 5, 7]
        flash_on = [bars[i].animate.set_fill(RED).set_stroke(RED) for i in flash_indices]
        flash_off = [bars[i].animate.set_fill(YELLOW).set_stroke(YELLOW) for i in flash_indices]
        
        self.play(*flash_on, run_time=0.4)
        self.play(*flash_off, run_time=0.4)
        self.play(*flash_on, run_time=0.4)
        self.play(*flash_off, run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Is there a hidden order within this universal randomness?"
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Fixed Issue 21: Moved question mark to B3 and adjusted scale to avoid overlap and improve visibility.
        question_mark = Text("?", color=YELLOW).scale(2)
        self.place_at_grid(question_mark, "B3", scale_factor=1.1)
        
        self.play(FadeIn(question_mark, shift=UP))
        self.wait(2)
        
        self.play(
            FadeOut(question_mark),
            FadeOut(bars),
            self.lecture[2].animate.set_color(WHITE)
        )
        self.wait(1)
