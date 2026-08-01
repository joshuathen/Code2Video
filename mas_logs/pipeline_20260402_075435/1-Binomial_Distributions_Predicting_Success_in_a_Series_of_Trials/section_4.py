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
        # 1. Setup layout with script content
        lecture_lines = [
            "We can visualize probabilities using a histogram.",
            "The bars show the likelihood of each outcome.",
            "Changing n or p shifts the distribution's shape.",
            "With p at 0.5, the distribution is perfectly symmetric.",
            "High or low p values create a skewed graph."
        ]
        self.setup_layout("Visualizing the Distribution (The Histogram)", lecture_lines)

        # 2. Define standard colors
        COLOR_GREY = "#808080"
        COLOR_CYAN = "#00FFFF"
        COLOR_RED = "#FF0000"
        COLOR_GREEN = "#00FF00"
        
        # 3. Mathematical helpers and State
        n = 10
        p_tracker = ValueTracker(0.5)

        def binomial_pmf(n, k, p):
            if k < 0 or k > n: return 0
            return comb(n, k) * (p**k) * ((1-p)**(n-k))

        # 4. Create Graphical Elements
        # White axes (#FFFFFF)
        axes = Axes(
            x_range=[0, 11, 1],
            y_range=[0, 0.5, 0.1],
            x_length=5,
            y_length=4,
            axis_config={"color": WHITE, "include_tip": False},
            tips=False
        )
        self.place_in_area(axes, "B3", "E6", scale_factor=0.9)

        # Axis labels 'k' and 'P(X=k)'
        k_label = Text("k", font_size=24, color=WHITE)
        self.place_at_grid(k_label, "F4", scale_factor=0.8)
        
        prob_label = Text("P(X=k)", font_size=24, color=WHITE).rotate(90 * DEGREES)
        self.place_at_grid(prob_label, "C2", scale_factor=0.8)

        # 5. Create Dynamic Histogram Bars
        bars = VGroup()
        for k in range(n + 1):
            prob = binomial_pmf(n, k, p_tracker.get_value())
            # Start with a very small height to avoid zeros during scaling
            h = max(prob * axes.y_axis.unit_size, 0.01)
            bar = Rectangle(
                width=0.35,
                height=h,
                fill_color=COLOR_GREY,
                fill_opacity=0.7,
                stroke_color=WHITE,
                stroke_width=1
            )
            bar.move_to(axes.c2p(k, 0), aligned_edge=DOWN)
            bars.add(bar)

        # Updater to adjust bar heights as p changes
        def update_bars(mob):
            current_p = p_tracker.get_value()
            for k, bar in enumerate(mob):
                prob = binomial_pmf(n, k, current_p)
                new_height = max(prob * axes.y_axis.unit_size, 0.01)
                bar.stretch_to_fit_height(new_height)
                bar.move_to(axes.c2p(k, 0), aligned_edge=DOWN)

        bars.add_updater(update_bars)

        # === Animation for Lecture Line 1 ===
        # "We can visualize probabilities using a histogram."
        self.play(self.lecture[0].animate.set_color(COLOR_GREY))
        self.play(Create(axes), Write(k_label), Write(prob_label))
        self.play(Create(bars), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The bars show the likelihood of each outcome."
        self.play(self.lecture[1].animate.set_color(COLOR_GREY))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # "Changing n or p shifts the distribution's shape."
        # The bars change heights smoothly to show different probability distributions.
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.play(p_tracker.animate.set_value(0.2), run_time=1.5)
        self.play(p_tracker.animate.set_value(0.8), run_time=1.5)
        self.play(p_tracker.animate.set_value(0.5), run_time=1.2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "With p at 0.5, the distribution is perfectly symmetric."
        # The bars align into a symmetric bell-shape in cyan (#00FFFF).
        self.play(self.lecture[3].animate.set_color(COLOR_CYAN))
        self.play(
            p_tracker.animate.set_value(0.5),
            bars.animate.set_color(COLOR_CYAN),
            run_time=1.2
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # "High or low p values create a skewed graph."
        # Left (peak on left, low p) in Red
        self.play(self.lecture[4].animate.set_color(COLOR_RED))
        self.play(
            p_tracker.animate.set_value(0.15),
            bars.animate.set_color(COLOR_RED),
            run_time=2
        )
        self.wait(1)
        
        # Right (peak on right, high p) in Green
        self.play(self.lecture[4].animate.set_color(COLOR_GREEN))
        self.play(
            p_tracker.animate.set_value(0.85),
            bars.animate.set_color(COLOR_GREEN),
            run_time=2
        )
        self.wait(2)
