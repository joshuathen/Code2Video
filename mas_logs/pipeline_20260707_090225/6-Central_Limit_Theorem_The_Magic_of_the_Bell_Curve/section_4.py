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

        # === Asset Preparation ===

        # 1. Skewed Population Distribution (Preview)
        # Positioned at A3-B5 (Issue 40 Fix)
        pop_axes = Axes(
            x_range=[0, 5], y_range=[0, 1],
            axis_config={"include_tip": False, "stroke_width": 1},
            x_length=2.5, y_length=1.5
        ).set_color(GRAY_C)
        
        # A skewed curve: Gamma-like 4*x*exp(-2x)
        pop_curve = pop_axes.plot(lambda x: 4 * x * np.exp(-2 * x), x_range=[0, 5], color=YELLOW)
        pop_label = Text("Skewed Population", font_size=16, color=WHITE).next_to(pop_axes, UP, buff=0.1)
        pop_group = VGroup(pop_axes, pop_curve, pop_label)
        self.place_in_area(pop_group, "A3", "B5", scale_factor=0.8)

        # 2. Sampling Distribution Main Axes
        # Positioned at C2-F6 (Issue 41 Fix)
        axes = Axes(
            x_range=[-3, 3, 1], y_range=[0, 1.5, 0.5],
            axis_config={"include_tip": False},
            x_length=5.5, y_length=3.0
        ).set_color(WHITE)
        
        axes_labels = VGroup(
            Text("Sample Mean", font_size=16).next_to(axes.x_axis, DOWN),
            Text("Freq", font_size=16).next_to(axes.y_axis, LEFT)
        )
        main_axes_group = VGroup(axes, axes_labels)
        self.place_in_area(main_axes_group, "C2", "F6", scale_factor=1.0)

        # 3. Dynamic Histogram Setup
        n_tracker = ValueTracker(2)
        num_bars = 20
        bar_width = 5.0 / num_bars
        bars = VGroup(*[
            Rectangle(
                width=bar_width * 0.8, height=0.1, 
                fill_opacity=0.5, fill_color=BLUE_B, stroke_width=0.5
            ) for _ in range(num_bars)
        ])

        def update_histogram(mob):
            n = n_tracker.get_value()
            alpha = np.clip((n - 2) / 28.0, 0, 1) # Transition factor from 2 to 30
            
            for i, bar in enumerate(mob):
                # Map index to x in range [-2.5, 2.5]
                x = -2.5 + i * (5.0 / (num_bars - 1))
                
                # Skewed component (shifted and scaled)
                skew_val = max(0, (x + 1.2) * np.exp(-(x + 1.2)) * 3.5)
                # Normal component
                norm_val = np.exp(-(x**2) / (2 * 0.6**2)) * 1.2
                
                # Combined height
                h = (1 - alpha) * skew_val + alpha * norm_val
                
                bar.stretch_to_fit_height(max(0.05, h))
                bar.move_to(axes.c2p(x, 0), aligned_edge=DOWN)

        bars.add_updater(update_histogram)

        # 4. n-Value Indicator
        # Positioned at B2 (Issue 39 Fix)
        n_label = Text("Sample Size n =", font_size=20, color=WHITE)
        n_val_text = Text("2", font_size=20, color=GREEN)
        n_display = VGroup(n_label, n_val_text).arrange(RIGHT, buff=0.2)
        self.place_at_grid(n_display, "B2", scale_factor=0.9)

        def update_n_text(mob):
            val = int(n_tracker.get_value())
            new_mob = Text(str(val), font_size=20, color=GREEN)
            mob.become(new_mob.move_to(mob))
        n_val_text.add_updater(update_n_text)

        # 5. Final Normal Curve (to be revealed)
        bell_curve = axes.plot(
            lambda x: np.exp(-(x**2) / (2 * 0.6**2)) * 1.2,
            x_range=[-2.5, 2.5], color="#00FFFF"
        ).set_stroke(width=4)

        # === Animation for Lecture Line 1 ===
        # The Central Limit Theorem reveals a hidden mathematical law.
        self.play(self.lecture[0].animate.set_color(colors[0]))
        self.play(FadeIn(pop_group), FadeIn(main_axes_group))
        self.add(bars, n_display)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # As sample size grows, the distribution shape transforms.
        self.play(self.lecture[1].animate.set_color(colors[1]))
        self.play(n_tracker.animate.set_value(15), run_time=2.5, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Uniform or skewed data eventually yields a bell curve.
        self.play(self.lecture[2].animate.set_color(colors[2]))
        self.play(n_tracker.animate.set_value(30), run_time=2.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This normal distribution is symmetrical and highly predictable.
        self.play(self.lecture[3].animate.set_color(colors[3]))
        self.play(Create(bell_curve), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Large samples always find order within the original mess.
        self.play(self.lecture[4].animate.set_color(colors[4]))
        
        # Grand Reveal: Flash the Bell Curve
        self.play(
            bell_curve.animate.set_stroke(color=WHITE, width=8),
            run_time=0.4
        )
        self.play(
            bell_curve.animate.set_stroke(color="#00FFFF", width=5),
            run_time=0.8
        )
        self.wait(2)

        # Cleanup updaters for safety
        bars.remove_updater(update_histogram)
        n_val_text.remove_updater(update_n_text)
