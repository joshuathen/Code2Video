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
        self.setup_layout("Prerequisite: The Bernoulli Trial", [
            "A Bernoulli trial has only two possible outcomes.",
            "We label these results as Success or Failure.",
            "The probability of success must stay the same."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show a card labeled 'Bernoulli Trial' in the center of the screen in white #FFFFFF.
        self.lecture[0].set_color(WHITE)
        
        card_rect = RoundedRectangle(corner_radius=0.2, width=4, height=2, color=WHITE)
        card_text = Text("Bernoulli Trial", color=WHITE, font_size=36)
        card = VGroup(card_rect, card_text)
        # Resolved Issue 22: Fixed card overlap by using B3 to E4
        self.place_in_area(card, 'B3', 'E4', scale_factor=0.8)
        
        self.play(FadeIn(card))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Split the screen: left side shows a fish icon 'Success (p=0.7)' in green #00FF00, 
        # right side shows 'Failure (q=0.3)' in red #FF0000.
        self.play(self.lecture[1].animate.set_color(GREEN))
        
        # Success side (Left part of grid)
        # Resolved Issue 19: Integrated fish icon asset
        success_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fish.svg")
        success_icon.set_color(GREEN)
        success_label = Text("Success (p=0.7)", color=GREEN, font_size=24)
        success_group = VGroup(success_icon, success_label).arrange(DOWN)
        # Resolved Issue 23: Fixed success_group overlap by using B1 to E2
        self.place_in_area(success_group, 'B1', 'E2', scale_factor=0.8)
        
        # Failure side (Right part of grid)
        failure_icon = Cross(stroke_width=6, color=RED)
        failure_label = Text("Failure (q=0.3)", color=RED, font_size=24)
        failure_group = VGroup(failure_icon, failure_label).arrange(DOWN)
        # Resolved Issue 24: Fixed failure_group overlap by using B5 to E6
        self.place_in_area(failure_group, 'B5', 'E6', scale_factor=0.8)

        self.play(
            FadeOut(card),
            FadeIn(success_group),
            FadeIn(failure_group)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Flash the 'p=0.7' label in yellow #FFFF00 to emphasize that the probability remains constant.
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        self.play(Flash(success_label, color=YELLOW, flash_radius=0.5))
        self.play(success_label.animate.set_color(YELLOW))
        self.wait(2)
