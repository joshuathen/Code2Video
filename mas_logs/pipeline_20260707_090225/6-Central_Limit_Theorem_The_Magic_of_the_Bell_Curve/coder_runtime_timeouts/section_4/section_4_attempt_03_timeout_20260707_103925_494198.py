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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup layout
        lines = [
            'The Central Limit Theorem reveals a hidden mathematical law.', 
            'As sample size grows, the distribution shape transforms.', 
            'Uniform or skewed data eventually yields a bell curve.', 
            'This normal distribution is symmetrical and highly predictable.', 
            'Large samples always find order within the original mess.'
        ]
        self.setup_layout("The Grand Reveal: The Bell Curve Emerges", lines)
        
        # Colors for lines
        colors = ["#FFFF00", "#00FF00", "#00FFFF", "#FF00FF", "#FFFFFF"]

        # === Pre-build Assets ===
        
        # 1. Skewed Population Preview (A4-B6 area)
        pop_axes = Axes(
            x_range=[0, 5], y_range=[0, 1], 
            axis_config={"include_tip": False, "stroke_width": 1},
            x_length=2.5, y_length=1.5
        ).set_color(GRAY_C)
        pop_curve = pop_axes.plot(lambda x: 4 * x * np.exp(-2 * x), x_range=[0, 5], color=YELLOW)
        pop_label = Text("Skewed Population", font_size=14, color=WHITE).next_to(pop_axes, UP, buff=0.1)
        pop_group = VGroup(pop_axes, pop_curve, pop_label)
        self.place_in_area(pop_group, "A4", "B6", scale_factor=0.8)

        # 2. Main Sampling Axes (C1-F6 area)
        axes = Axes(
            x_range=[-4, 4, 1], y_range=[0, 1.2, 0.2],
            axis_config={"include_tip": False},
            x_length=5, y_length=3.5
        ).set_color(WHITE)
        
        # FIX: Replacing LaTeX strings with Text objects to avoid FileNotFoundError: 'latex'
        axes_labels = axes.get_axis_labels(
            x_label=Text("Sample Mean", font_size=18), 
            y_label=Text("Frequency", font_size=18)
        )
        
        main_display = VGroup(axes, axes_labels)
        self.place_in_area(main_display, "C1", "F6", scale_factor=0.9)

        # 3. Dynamic Histogram Setup
        n_tracker = ValueTracker(2)
        num_bars = 24
        bar_width = 4.0 / num_bars
        bars = VGroup(*[
            Rectangle(
                width=bar_width * 0.9, height=0.1, 
                fill_opacity=0.6, fill_color=BLUE, stroke_width=1
            ) for _ in range(num_bars)
        ])
        
        def update_bars(m):
            n = n_tracker.get_value()
            sigma = 1.8 / np.sqrt(n) 
            mu = 0
            for i, bar in enumerate(m):
                x = -2 + i * (4/num_bars)
                h = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x/sigma)**2)
                h = min(h, 1.1)
                bar.set_height(max(0.01, h * 2.8), stretch=True) 
                bar.move_to(axes.c2p(x, 0), aligned_edge=DOWN)

        bars.add_updater(update_bars)

        # 4. n-Value Label
        n_label = Text("n = ", font_size=20, color=WHITE)
        n_value = DecimalNumber(2, num_decimal_places=0, font_size=20, color=GREEN, mob_class=Text)
        n_display = VGroup(n_label, n_value).arrange(RIGHT, buff=0.1)
        self.place_at_grid(n_display, "C1", scale_factor=1.0)
        n_value.add_updater(lambda d: d.set_value(n_tracker.get_value()))

        # 5. Normal Curve (Cyan)
        bell_curve = axes.plot(
            lambda x: (1.0 / (0.33 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x/0.33)**2),
            x_range=[-3, 3], color="#00FFFF"
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        self.play(Create(pop_group), run_time=1)
        self.play(Create(axes), Create(axes_labels), FadeIn(bars), FadeIn(n_display))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        self.play(n_tracker.animate.set_value(15), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        self.play(n_tracker.animate.set_value(30), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(colors[3]))
        self.play(Create(bell_curve), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(colors[4]))
        self.play(
            bell_curve.animate.set_stroke(color="#FFFFFF", width=8),
            run_time=0.3
        )
        self.play(
            bell_curve.animate.set_stroke(color="#00FFFF", width=4),
            run_time=0.7
        )
        self.wait(2)

        # Cleanup updaters
        bars.remove_updater(update_bars)
        n_value.remove_updater(None)