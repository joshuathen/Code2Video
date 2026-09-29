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
        lecture_lines = ["PMF shape changes with n and p.", "Observe how bar heights shift with probability.", "The sum of all probabilities always equals one."]
        self.setup_layout("Visualizing the Probability Mass Function", lecture_lines)
        
        # Define chart
        def get_bars(n, p):
            x_vals = np.arange(n + 1)
            probs = binom.pmf(x_vals, n, p)
            bars = VGroup(*[Rectangle(height=prob*3, width=0.4, color=BLUE, fill_opacity=0.7) for prob in probs])
            bars.arrange(RIGHT, aligned_edge=DOWN, buff=0.1)
            return bars

        chart = get_bars(10, 0.5)
        self.place_in_area(chart, 'B3', 'E6', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(chart))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        chart_p_low = get_bars(10, 0.2)
        self.place_in_area(chart_p_low, 'B3', 'E6', scale_factor=0.7)
        self.play(Transform(chart, chart_p_low))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        # Total probability indicator
        total_text = Text("Sum = 1.0", font_size=24, color=GREEN)
        self.place_at_grid(total_text, 'E4', scale_factor=0.8)
        self.play(FadeIn(total_text))
        self.wait(1)
        self.play(FadeOut(chart), FadeOut(total_text), self.lecture[2].animate.set_color(WHITE))
