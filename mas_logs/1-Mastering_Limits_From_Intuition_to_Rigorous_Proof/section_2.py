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

class Section2Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        self.setup_layout("The Epsilon-Delta (ε-δ) Definition: The Precision Game", [
            "Rigorous limits use epsilon for vertical error tolerance.",
            "We define a horizontal input range called delta.",
            "For any epsilon, we must find a valid delta.",
            "This creates a target box around the limit.",
            "If the function stays inside, the limit is proven."
        ])

        # Colors
        COLOR_EPS = "#FFFFE0"  # Yellow
        COLOR_DELTA = "#90EE90" # Green
        COLOR_BOX = "#FFD700"   # Gold/Target Box
        COLOR_LOGIC = "#E0FFFF" # Cyan
        
        # Shared elements
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=4,
            y_length=3,
            axis_config={"include_tip": True}
        )
        self.place_in_area(axes, "C2", "F6", scale_factor=0.9)
        
        func = axes.plot(lambda x: 0.5 * x + 1, x_range=[0, 4], color=WHITE)
        point_c_l = Dot(axes.c2p(2, 2), color=RED)
        # Fixed: Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        label_c_l = Text("(c, L)", font_size=20).next_to(point_c_l, UR, buff=0.1)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_EPS)
        
        # Fixed: Use VGroup of Text to mimic MathTex grouping for coloring
        formula = VGroup(
            Text("|f(x) - L| < ε", font_size=24),
            Text(" if ", font_size=24),
            Text("0 < |x - c| < δ", font_size=24)
        ).arrange(RIGHT, buff=0.1)
        
        formula[0].set_color(COLOR_EPS)
        formula[2].set_color(COLOR_DELTA)
        self.place_at_grid(formula, "A4", scale_factor=1.0)
        
        eps_val = 0.4
        # Fixed: Corrected invalid method name axes.get_horizontal_line_delta_z
        eps_band_line = axes.get_horizontal_line(axes.c2p(4, 2 + eps_val), color=COLOR_EPS)
        
        # Manual bands for better control
        y_top = axes.c2p(0, 2 + eps_val)[1]
        y_bottom = axes.c2p(0, 2 - eps_val)[1]
        x_start = axes.c2p(0, 0)[0]
        x_end = axes.c2p(4, 0)[0]
        
        h_region = Rectangle(
            width=x_end - x_start,
            height=y_top - y_bottom,
            fill_color=COLOR_EPS,
            fill_opacity=0.2,
            stroke_width=0
        ).move_to(axes.c2p(2, 2))
        
        # Fixed: Replaced MathTex with Text for labels
        label_eps_p = Text("L+ε", color=COLOR_EPS, font_size=20)
        label_eps_m = Text("L-ε", color=COLOR_EPS, font_size=20)
        self.place_at_grid(label_eps_p, "D1", scale_factor=1.0)
        self.place_at_grid(label_eps_m, "E1", scale_factor=1.0)

        self.play(Create(axes), Create(func))
        self.play(FadeIn(point_c_l, label_c_l), Write(formula))
        self.play(FadeIn(h_region), Write(label_eps_p), Write(label_eps_m))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_DELTA)
        
        delta_val = 0.8 # delta = 2 * epsilon for slope 0.5
        x_left = axes.c2p(2 - delta_val, 0)[0]
        x_right = axes.c2p(2 + delta_val, 0)[0]
        y_start = axes.c2p(0, 0)[1]
        y_end = axes.c2p(0, 4)[1]
        
        v_region = Rectangle(
            width=x_right - x_left,
            height=y_end - y_start,
            fill_color=COLOR_DELTA,
            fill_opacity=0.2,
            stroke_width=0
        ).move_to(axes.c2p(2, 2))
        
        # Fixed: Replaced MathTex with Text for labels
        label_delta_p = Text("c+δ", color=COLOR_DELTA, font_size=20)
        label_delta_m = Text("c-δ", color=COLOR_DELTA, font_size=20)
        # delta labels at bottom
        self.place_at_grid(label_delta_m, "F3", scale_factor=1.0)
        self.place_at_grid(label_delta_p, "F5", scale_factor=1.0)

        self.play(FadeIn(v_region), Write(label_delta_p), Write(label_delta_m))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_LOGIC)
        self.play(Indicate(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_BOX)
        
        target_box = Rectangle(
            width=x_right - x_left,
            height=y_top - y_bottom,
            stroke_color=COLOR_BOX,
            stroke_width=2,
            fill_color=COLOR_BOX,
            fill_opacity=0.3
        ).move_to(axes.c2p(2, 2))
        
        self.play(Create(target_box))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        
        # Setup ValueTracker for shrinking
        tracker = ValueTracker(0.4)
        
        # Updaters
        def update_h_region(m):
            e = tracker.get_value()
            y_t = axes.c2p(0, 2 + e)[1]
            y_b = axes.c2p(0, 2 - e)[1]
            m.stretch_to_fit_height(max(0.01, y_t - y_b))
            m.move_to(axes.c2p(2, 2))

        def update_v_region(m):
            e = tracker.get_value()
            d = 2 * e
            x_l = axes.c2p(2 - d, 0)[0]
            x_r = axes.c2p(2 + d, 0)[0]
            m.stretch_to_fit_width(max(0.01, x_r - x_l))
            m.move_to(axes.c2p(2, 2))
            
        def update_box(m):
            e = tracker.get_value()
            d = 2 * e
            y_t = axes.c2p(0, 2 + e)[1]
            y_b = axes.c2p(0, 2 - e)[1]
            x_l = axes.c2p(2 - d, 0)[0]
            x_r = axes.c2p(2 + d, 0)[0]
            m.stretch_to_fit_height(max(0.01, y_t - y_b))
            m.stretch_to_fit_width(max(0.01, x_r - x_l))
            m.move_to(axes.c2p(2, 2))

        h_region.add_updater(update_h_region)
        v_region.add_updater(update_v_region)
        target_box.add_updater(update_box)
        
        self.play(tracker.animate.set_value(0.15), run_time=3)
        self.wait(2)
