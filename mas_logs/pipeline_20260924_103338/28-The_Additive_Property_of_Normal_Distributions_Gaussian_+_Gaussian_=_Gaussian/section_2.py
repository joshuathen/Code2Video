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
        lecture_lines = [
            "Adding variables is essentially a convolution of PDFs.",
            "The combined density function represents total uncertainty.",
            "This process smooths and widens the resulting curve."
        ]
        self.setup_layout("The Core Concept: Convolution", lecture_lines)
        
        # Visuals: Convolution demo
        array1 = VGroup(*[Square(side_length=0.5, color=BLUE).set_fill(BLUE, opacity=0.3) for _ in range(5)]).arrange(RIGHT, buff=0.1)
        array2 = VGroup(*[Square(side_length=0.5, color=GREEN).set_fill(GREEN, opacity=0.3) for _ in range(3)]).arrange(RIGHT, buff=0.1)
        result = VGroup(*[Square(side_length=0.5, color=YELLOW).set_fill(YELLOW, opacity=0.3) for _ in range(7)]).arrange(RIGHT, buff=0.1)
        
        # Asset: Sliding Window
        window = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/window.svg", color=WHITE)
        
        self.place_in_area(array1, 'B2', 'B4', scale_factor=0.7)
        self.place_in_area(array2, 'C2', 'C4', scale_factor=0.7)
        self.place_in_area(result, 'E2', 'E5', scale_factor=0.8)
        
        sum_symbol = MathTex(r"\\sum", color="#00FFFF", font_size=48)
        self.place_at_grid(sum_symbol, 'D3', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(FadeIn(array1), FadeIn(array2))
        self.place_at_grid(window, 'B2', scale_factor=0.5)
        self.play(FadeIn(window))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        self.play(FadeIn(sum_symbol))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(FadeIn(result))
        self.wait(2)
