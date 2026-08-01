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

class Section5Scene(TeachingScene):
    def construct(self):
        # Lecture lines for setup
        lines = [
            "Watch the point move along the curve.",
            "Height and steepness stay perfectly equal.",
            "This invariant property is calculus's 'holy grail.'",
            "It simplifies calculations in physics and finance.",
            "e to the x is its own shadow."
        ]
        self.setup_layout("Visualizing the 'Invariant' Property", lines)

        # Define Colors
        GOLD = "#FFD700"
        BLUE = "#0000FF"
        GREEN = "#00FF00"
        WHITE_C = "#FFFFFF"

        # ValueTracker for dynamic animation
        x_tracker = ValueTracker(-2.0)

        # 1. Plot the function f(x) = e^x in gold
        axes = Axes(
            x_range=[-2, 2.1, 1],
            y_range=[0, 8.5, 2],
            x_length=3.5,
            y_length=4.5,
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.place_in_area(axes, "A1", "F4", scale_factor=1.0)
        
        graph = axes.plot(lambda x: np.exp(x), x_range=[-2, 2.05], color=GOLD)
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        graph_label = Text("f(x) = e^x", color=GOLD, font_size=24)
        self.place_at_grid(graph_label, "A1", scale_factor=1)

        # 2. Position two vertical bars on the right
        # Height Bar (Blue)
        h_bar = Rectangle(width=0.5, height=0.1, fill_opacity=1.0, color=BLUE, stroke_width=0)
        h_label = Text("Height", font_size=18, color=BLUE)
        self.place_at_grid(h_label, "D5", scale_factor=0.8)
        
        # Slope Bar (Green)
        s_bar = Rectangle(width=0.5, height=0.1, fill_opacity=1.0, color=GREEN, stroke_width=0)
        s_label = Text("Slope", font_size=18, color=GREEN)
        self.place_at_grid(s_label, "D6", scale_factor=0.8)

        # Bar base coordinates (near bottom of right grid)
        h_base = self.grid["F5"]
        s_base = self.grid["F6"]

        # 3. Moving point along the curve
        dot = Dot(color=WHITE_C, radius=0.08)
        dot.add_updater(lambda d: d.move_to(axes.c2p(x_tracker.get_value(), np.exp(x_tracker.get_value()))))
        
        # 4. Updaters for the two bars to match y-value
        def update_bar_to_val(bar, base, val):
            # Scale val to fit visual area (height of area is approx 5 units)
            # max val is e^2 ~ 7.38. We scale by 0.45 to keep it under 4 units.
            h = max(0.05, val * 0.45)
            bar.stretch_to_fit_height(h, about_edge=DOWN)
            bar.move_to(base, aligned_edge=DOWN)

        h_bar.add_updater(lambda b: update_bar_to_val(b, h_base, np.exp(x_tracker.get_value())))
        s_bar.add_updater(lambda b: update_bar_to_val(b, s_base, np.exp(x_tracker.get_value())))

        # 5. Flash Text "Height = Slope"
        flash_text = Text("Height = Slope", font_size=24, color=WHITE_C)
        self.place_in_area(flash_text, "A2", "A4", scale_factor=0.9)
        flash_text.set_opacity(0)

        # --- ANIMATION SEQUENCES ---

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(GOLD)
        self.add(axes, graph, graph_label, dot)
        self.play(x_tracker.animate.set_value(-0.5), run_time=2, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.add(h_bar, s_bar, h_label, s_label)
        self.play(x_tracker.animate.set_value(0.0), run_time=1.5, rate_func=linear)
        # Flash at integer x=0
        self.play(flash_text.animate.set_opacity(1), run_time=0.3)
        self.play(flash_text.animate.set_opacity(0), run_time=0.3)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE_C)
        self.play(x_tracker.animate.set_value(1.0), run_time=1.5, rate_func=linear)
        # Flash at integer x=1
        self.play(flash_text.animate.set_opacity(1), run_time=0.3)
        self.play(flash_text.animate.set_opacity(0), run_time=0.3)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN)
        self.play(x_tracker.animate.set_value(2.0), run_time=2, rate_func=linear)
        # Flash at integer x=2
        self.play(flash_text.animate.set_opacity(1), run_time=0.3)
        self.play(flash_text.animate.set_opacity(0), run_time=0.3)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GOLD)
        # Sweep back to show consistency
        self.play(x_tracker.animate.set_value(-1.5), run_time=3)
        self.wait(2)
