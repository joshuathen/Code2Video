from manim import *
import numpy as np
from math import comb

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
        # Initial Setup
        lines = [
            'Plotting these probabilities creates a visual distribution.',
            'The peak shifts based on the success probability.',
            'Increasing trials makes the histogram look like a mountain.',
            'The shape eventually becomes a smooth bell curve.',
            'This is known as the Normal approximation.'
        ]
        self.setup_layout("The Shape of Probability: Histograms", lines)

        def get_binomial_pmf(n, p):
            return [comb(n, k) * (p**k) * ((1-p)**(n-k)) for k in range(n + 1)]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#58C4DD")
        
        # Display Formula and Params using Text
        formula_tex = Text(
            "P(X=k) = C(n,k) * p^k * (1-p)^(n-k)",
            font_size=28, color=WHITE
        )
        self.place_in_area(formula_tex, 'A1', 'A3', scale_factor=0.8)
        
        params_text = Text(
            "n=10, p=0.5",
            font_size=32, color="#58C4DD"
        )
        self.place_in_area(params_text, 'A4', 'A6', scale_factor=0.7)
        
        # Initial Chart n=10, p=0.5
        probs_n10_p05 = get_binomial_pmf(10, 0.5)
        chart_active = BarChart(
            values=probs_n10_p05,
            y_range=[0, 0.3, 0.1],
            bar_colors=["#58C4DD"],
            x_length=6,
            y_length=4,
            axis_config={"font_size": 24, "label_constructor": Text}
        )
        self.place_in_area(chart_active, 'B1', 'F6', scale_factor=0.6)
        
        self.play(Write(formula_tex), Write(params_text))
        self.play(Create(chart_active))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#58C4DD")
        
        # Shift p to 0.2
        params_p02 = Text("n=10, p=0.2", font_size=32, color="#58C4DD")
        self.place_in_area(params_p02, 'A4', 'A6', scale_factor=0.7)
        
        probs_n10_p02 = get_binomial_pmf(10, 0.2)
        chart_n10_p02 = BarChart(
            values=probs_n10_p02,
            y_range=[0, 0.3, 0.1],
            bar_colors=["#58C4DD"],
            x_length=6,
            y_length=4,
            axis_config={"font_size": 24, "label_constructor": Text}
        )
        self.place_in_area(chart_n10_p02, 'B1', 'F6', scale_factor=0.6)
        
        self.play(Transform(params_text, params_p02), Transform(chart_active, chart_n10_p02))
        self.wait(1)

        # Shift p to 0.8
        params_p08 = Text("n=10, p=0.8", font_size=32, color="#58C4DD")
        self.place_in_area(params_p08, 'A4', 'A6', scale_factor=0.7)
        
        probs_n10_p08 = get_binomial_pmf(10, 0.8)
        chart_n10_p08 = BarChart(
            values=probs_n10_p08,
            y_range=[0, 0.3, 0.1],
            bar_colors=["#58C4DD"],
            x_length=6,
            y_length=4,
            axis_config={"font_size": 24, "label_constructor": Text}
        )
        self.place_in_area(chart_n10_p08, 'B1', 'F6', scale_factor=0.6)
        
        self.play(Transform(params_text, params_p08), Transform(chart_active, chart_n10_p08))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#58C4DD")
        params_n50 = Text("n=50, p=0.5", font_size=32, color="#58C4DD")
        self.place_in_area(params_n50, 'A4', 'A6', scale_factor=0.7)
        
        probs_n50 = get_binomial_pmf(50, 0.5)
        chart_n50 = BarChart(
            values=probs_n50,
            y_range=[0, 0.15, 0.05],
            bar_colors=["#58C4DD"],
            x_length=6,
            y_length=4,
            axis_config={"font_size": 24, "label_constructor": Text}
        )
        self.place_in_area(chart_n50, 'B1', 'F6', scale_factor=0.6)
        
        self.play(Transform(params_text, params_n50), Transform(chart_active, chart_n50))
        self.wait(1)

        # === Animation for Lecture Line 4 & 5 ===
        self.lecture[3].set_color("#58C4DD")
        self.wait(0.5)
        self.lecture[4].set_color("#58C4DD")
        
        approx_label = Text("Normal Approximation", font_size=32, color="#C9C227")
        self.place_in_area(approx_label, 'A1', 'A3', scale_factor=0.8)
        
        self.play(Transform(formula_tex, approx_label))
        self.wait(2)
