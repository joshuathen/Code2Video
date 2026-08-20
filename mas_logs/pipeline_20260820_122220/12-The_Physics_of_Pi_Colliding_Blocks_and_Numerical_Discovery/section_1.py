from manim import *
import numpy as np
import os

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
        lecture_lines = [
            "Can colliding blocks calculate digits of Pi?",
            "Consider two blocks sliding on a frictionless surface.",
            "The mass ratio determines the collision dynamics."
        ]
        self.setup_layout("The Unexpected Appearance of Pi", lecture_lines)
        
        # Load assets
        surface = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/surface.svg")
        blocks = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg")
        blocks.set_color("#FF0000")
        
        # Positioning using fixed grid positions as requested in feedback
        self.place_at_grid(surface, "D6", scale_factor=1.0)
        self.place_at_grid(blocks, "D4", scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.play(DrawBorderThenFill(surface), FadeIn(blocks))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(GRAY), FadeIn(self.lecture[1]))
        self.play(blocks.animate.shift(LEFT * 1.5))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(GRAY), FadeIn(self.lecture[2]))
        
        label_a = Text("1kg", font_size=16).scale(0.7)
        label_b = Text("100kg", font_size=16).scale(0.7)
        label_a.next_to(blocks, UP, buff=0.1)
        label_b.next_to(blocks, DOWN, buff=0.1)
        
        self.play(FadeIn(label_a), FadeIn(label_b))
        self.wait(2)
