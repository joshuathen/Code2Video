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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application: Why it Matters", [
            "Fourier transforms are essential for noise cancellation.",
            "Compression algorithms use frequencies to store images.",
            "Technology relies on splitting signals into components."
        ])
        
        # Assets
        headphones = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/headphones.svg")
        computer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/computer.svg")
        
        # Create signals
        axes = Axes(x_range=[0, 4, 1], y_range=[-2, 2, 1], axis_config={"include_numbers": False}).scale(0.5)
        
        noisy_func = lambda x: np.sin(2 * np.pi * x) + 0.5 * np.sin(10 * np.pi * x)
        clean_func = lambda x: np.sin(2 * np.pi * x)
        
        noisy_graph = axes.plot(noisy_func, color=RED)
        clean_graph = axes.plot(clean_func, color=GREEN)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(RED)
        self.place_in_area(axes, 'A4', 'C6', scale_factor=0.6)
        self.place_at_grid(headphones, 'A1', scale_factor=0.3)
        self.add(noisy_graph, headphones)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        self.place_at_grid(computer, 'C1', scale_factor=0.3)
        self.play(
            Transform(noisy_graph, clean_graph),
            FadeIn(computer)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        # Highlight signals by showing them apart
        separated = VGroup(
            axes.copy(),
            axes.plot(lambda x: 0.5 * np.sin(10 * np.pi * x), color=RED).scale(0.8)
        ).arrange(DOWN).scale(0.5)
        self.place_in_area(separated, 'D4', 'F6', scale_factor=0.6)
        self.play(FadeIn(separated))
        self.wait(2)
