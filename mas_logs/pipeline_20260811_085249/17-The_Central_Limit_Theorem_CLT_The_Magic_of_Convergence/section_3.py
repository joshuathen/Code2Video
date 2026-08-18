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
        lecture_lines = [
            "Sampling distributions track the average of sample means.",
            "Even non-normal populations produce orderly sample means.",
            "These means gradually cluster into a bell curve."
        ]
        self.setup_layout("The Core Concept: Sampling Distributions", lecture_lines)
        
        # Assets
        urn = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/urn.svg")
        scoop = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scoop.svg")
        
        # Population histogram
        pop_hist = BarChart([1, 4, 8, 4, 1], bar_names=["A", "B", "C", "D", "E"], y_range=[0, 10, 2])
        self.place_in_area(pop_hist, 'A4', 'C6', scale_factor=0.6)
        
        # Means storage
        sample_means = VGroup()

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_at_grid(urn, 'B2', scale_factor=0.5)
        self.play(FadeIn(urn), Create(pop_hist))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        # Simulate sampling
        for _ in range(5):
            dot = Dot(color=BLUE).move_to(pop_hist.get_center())
            self.add(dot)
            sample_means.add(dot)
            self.play(dot.animate.shift(RIGHT * 1 + UP * (np.random.random()-0.5)), run_time=0.3)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.place_at_grid(scoop, 'E2', scale_factor=0.5)
        curve = FunctionGraph(lambda x: 3 * np.exp(-x**2), x_range=[-2, 2], color=GREEN)
        self.place_in_area(curve, 'D2', 'F5', scale_factor=0.7)
        self.play(FadeIn(scoop), Create(curve), FadeTransform(sample_means, curve))
        
        self.wait(2)
