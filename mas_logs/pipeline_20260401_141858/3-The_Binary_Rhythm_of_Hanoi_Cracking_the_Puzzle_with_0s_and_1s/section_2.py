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
        # Setup title and lecture lines
        lecture_lines = [
            "Binary numbers count the steps of our puzzle solution.",
            "At step one, the rightmost bit flips to one.",
            "At step two, the second bit is the rightmost flip."
        ]
        self.setup_layout("Prerequisite: Binary Counting as a Pulse", lecture_lines)

        # Colors
        HIGHLIGHT_COLOR = "#FFFF00"  # Yellow
        NORMAL_COLOR = "#FFFFFF"     # White

        # Prepare binary number list (000 to 111)
        # Each number is a VGroup of 3 digits
        binary_list = VGroup(*[
            VGroup(*[Text(char, font_size=36, color=NORMAL_COLOR) for char in bin(i)[2:].zfill(3)]).arrange(RIGHT, buff=0.1)
            for i in range(8)
        ]).arrange(DOWN, buff=0.3)
        
        # Position the list in the area B3 to F5 (Issue 31, 32, 46)
        self.place_in_area(binary_list, "B3", "F5", scale_factor=1.1)

        # Icons (Issue 25, 46)
        steps_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/steps.svg")
        self.place_at_grid(steps_icon, "B2", scale_factor=0.6)
        
        puzzle_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/puzzle.svg")
        self.place_at_grid(puzzle_icon, "C6", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        # Binary numbers count the steps of our puzzle solution.
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        self.play(
            Create(binary_list),
            FadeIn(steps_icon),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # At step one, the rightmost bit flips to one.
        self.play(
            self.lecture[0].animate.set_color(NORMAL_COLOR),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Highlight transition from 000 (binary_list[0]) to 001 (binary_list[1])
        # The last bit of 001 flips (0 -> 1)
        rect1 = SurroundingRectangle(binary_list[1][2], color=HIGHLIGHT_COLOR, buff=0.1)
        self.play(Create(rect1))
        self.play(binary_list[1][2].animate.set_color(HIGHLIGHT_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # At step two, the second bit is the rightmost flip.
        self.play(
            self.lecture[1].animate.set_color(NORMAL_COLOR),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Transition from 001 (binary_list[1]) to 010 (binary_list[2])
        # Highlight the second bit as the rightmost change
        rect2 = SurroundingRectangle(binary_list[2][1], color=HIGHLIGHT_COLOR, buff=0.1)
        self.play(
            ReplacementTransform(rect1, rect2),
            FadeIn(puzzle_icon)
        )
        self.play(binary_list[2][1].animate.set_color(HIGHLIGHT_COLOR))
        self.wait(2)
