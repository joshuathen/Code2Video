from manim import *
import numpy as np
from scipy.stats import binom

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
        self.setup_layout("Visualizing the Probability Distribution", [
            "- Bar charts visualize binomial distributions.",
            "- Adjust parameters n and p.",
            "- Symmetry increases as n grows."
        ])

        def get_bar_chart(n, p):
            x = np.arange(0, n + 1)
            y = binom.pmf(x, n, p)
            # Create a simple Axes for the chart
            axes = Axes(x_range=[0, n, 1], y_range=[0, max(y)*1.1, 0.1], x_length=4, y_length=2)
            bars = VGroup(*[
                Rectangle(height=y_val * 2 / max(y), width=3.5 / (n + 1), color=BLUE_B, fill_opacity=0.7)
                for y_val in y
            ])
            bars.arrange(RIGHT, aligned_edge=DOWN, buff=0.1)
            return VGroup(axes, bars)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE_B))
        chart1 = get_bar_chart(5, 0.5)
        # Fixing Issue 34: Position binomial chart
        self.place_in_area(chart1, 'A3', 'D5', scale_factor=0.6)
        self.play(FadeIn(chart1))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW_A))
        chart2 = get_bar_chart(10, 0.5)
        self.place_in_area(chart2, 'A3', 'D5', scale_factor=0.6)
        self.play(Transform(chart1, chart2))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN_A))
        chart3 = get_bar_chart(20, 0.5)
        self.place_in_area(chart3, 'A3', 'D5', scale_factor=0.6)
        self.play(Transform(chart1, chart3))
        self.wait(2)
