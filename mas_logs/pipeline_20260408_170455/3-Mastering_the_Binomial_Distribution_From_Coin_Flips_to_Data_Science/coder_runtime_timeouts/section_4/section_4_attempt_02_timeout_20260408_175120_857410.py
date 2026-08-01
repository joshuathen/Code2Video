from manim import *
import math
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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initialization
        lines = [
            'Plotting these probabilities creates a visual distribution.',
            'The peak shifts based on the success probability.',
            'Increasing trials makes the histogram look like a mountain.',
            'The shape eventually becomes a smooth bell curve.',
            'This is known as the Normal approximation.'
        ]
        self.setup_layout("The Shape of Probability: Histograms", lines)

        def binom_pmf(n, p, k):
            if k < 0 or k > n:
                return 0
            try:
                return math.comb(n, k) * (p**k) * ((1-p)**(n-k))
            except OverflowError:
                return 0

        # Trackers for interactive animation
        p_track = ValueTracker(0.5)
        n_track = ValueTracker(10)

        # Create Axes - centered in the right area
        # Use label_constructor=Text to avoid dependency on a LaTeX installation
        axes = Axes(
            x_range=[0, 1.0, 0.25],
            y_range=[0, 0.4, 0.1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": False, "font_size": 18}
        ).add_coordinates(label_constructor=Text)
        
        self.place_in_area(axes, "A1", "F6")

        # Labels
        x_axis_label = Text("Success Ratio (k/n)", font_size=16).next_to(axes.x_axis, DOWN, buff=0.4)
        y_axis_label = Text("Probability", font_size=16).rotate(90*DEGREES).next_to(axes.y_axis, LEFT, buff=0.3)

        # Pre-create a VGroup of 101 rectangles to handle n from 10 to 100
        bars = VGroup(*[
            Rectangle(
                fill_color="#58C4DD",
                fill_opacity=0.7,
                stroke_width=0.5
            ) for _ in range(101)
        ])

        def update_bars(m):
            n_val = int(n_track.get_value())
            p_val = p_track.get_value()
            unit_width_pixels = axes.x_axis.get_unit_size() / n_val
            
            for k in range(101):
                if k <= n_val:
                    prob = binom_pmf(n_val, p_val, k)
                    h_pixels = prob * axes.y_axis.get_unit_size()
                    
                    m[k].set_visible(True)
                    m[k].set_width(max(unit_width_pixels * 0.9, 0.001), stretch=True)
                    m[k].set_height(max(h_pixels, 0.001), stretch=True)
                    m[k].move_to(axes.c2p(k/n_val, prob/2))
                    
                    if n_val > 40:
                        m[k].set_stroke(width=0.1)
                    else:
                        m[k].set_stroke(width=0.5)
                else:
                    m[k].set_visible(False)

        bars.add_updater(update_bars)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#58C4DD"))
        self.add(axes, x_axis_label, y_axis_label, bars)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#58C4DD"))
        self.play(p_track.animate.set_value(0.2), run_time=2)
        self.wait(0.5)
        self.play(p_track.animate.set_value(0.8), run_time=2)
        self.wait(0.5)
        self.play(p_track.animate.set_value(0.5), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#58C4DD"))
        self.play(n_track.animate.set_value(100), run_time=5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFFFF"))
        
        def normal_approx_func(x):
            n = 100
            p = 0.5
            q = 1 - p
            sigma_k = math.sqrt(n * p * q)
            mu_k = n * p
            k = x * n
            return (1 / (sigma_k * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((k - mu_k) / sigma_k)**2)

        curve = axes.plot(normal_approx_func, x_range=[0, 1], color="#FFFFFF", stroke_width=4)
        self.play(Create(curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        
        p_final = p_track.get_value()
        prob_at_peak = binom_pmf(100, p_final, int(100 * p_final))
        
        mean_line = Line(
            axes.c2p(p_final, 0),
            axes.c2p(p_final, prob_at_peak + 0.05),
            color="#FFFF00",
            stroke_width=4
        )
        
        # Replacing MathTex with Text to avoid FileNotFoundError: 'latex'
        mean_label = Text("n * p", color="#FFFF00", font_size=24)
        mean_label.next_to(mean_line, UP, buff=0.1)
        
        self.play(Create(mean_line), Write(mean_label))
        self.wait(4)