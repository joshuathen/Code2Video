from manim import *
import numpy as np
import random

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

class Section8Scene(TeachingScene):
    def construct(self):
        # Initializing Layout
        self.setup_layout(
            "Summary: The Bridge to Statistics",
            [
                "Start with any distribution, no matter how chaotic.",
                "Build a bridge toward structured scientific prediction.",
                "Ensure sample size is at least thirty.",
                "The results will always follow a bell curve.",
                "Statistics transforms chaos into clear, actionable order."
            ]
        )

        # Colors for mapping lines to animations
        COLOR_CHAOS = "#FF5555"
        COLOR_BRIDGE = "#FFFFFF"
        COLOR_STEPS = "#55AAFF"
        COLOR_NORMAL = "#FFD700"
        COLOR_ORDER = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        # "Start with any distribution, no matter how chaotic."
        self.lecture[0].set_color(COLOR_CHAOS)
        
        # Chaos Group: Random primitive shapes
        # Seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        chaos_elements = VGroup()
        for _ in range(15):
            shape = random.choice([
                Circle(radius=0.1, color=COLOR_CHAOS, fill_opacity=0.6),
                Square(side_length=0.2, color=COLOR_CHAOS, fill_opacity=0.6),
                Triangle(color=COLOR_CHAOS, fill_opacity=0.6).scale(0.12)
            ])
            # Spread shapes around a relative center
            shape.shift(np.random.uniform(-0.5, 0.5, 3))
            chaos_elements.add(shape)
        
        # Fix for Issue 54: Position chaos_elements at area B2 to D3 to avoid overlap with lecture area.
        self.place_in_area(chaos_elements, "B2", "D3", scale_factor=1.0)
        
        # Faint bell curve on the right representing the goal
        axes = Axes(
            x_range=[-3, 3], y_range=[0, 0.5], 
            axis_config={"include_tip": False, "stroke_width": 1},
            tips=False
        ).set_color(GRAY)
        bell_curve_plot = axes.plot(lambda x: 0.4 * np.exp(-x**2 / 2), color=COLOR_NORMAL, stroke_opacity=0.3)
        bell_group = VGroup(axes, bell_curve_plot)
        self.place_in_area(bell_group, "B5", "D6", scale_factor=0.6)

        self.play(Create(chaos_elements), FadeIn(bell_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Build a bridge toward structured scientific prediction."
        self.lecture[1].set_color(COLOR_BRIDGE)
        
        # Bridge connecting Chaos area and Normal area
        # Calculate centers manually to ensure clear connection
        bridge_start = (self.grid["B2"] + self.grid["D3"]) / 2
        bridge_end = (self.grid["B5"] + self.grid["D6"]) / 2
        bridge = Line(bridge_start, bridge_end, color=COLOR_BRIDGE, stroke_width=6)
        
        self.play(Create(bridge))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Ensure sample size is at least thirty."
        self.lecture[2].set_color(COLOR_STEPS)
        
        step1 = Text("1. Any Data", font_size=20, color=COLOR_STEPS)
        step2 = Text("2. n \u2265 30", font_size=20, color=COLOR_STEPS)
        step3 = Text("3. Normal Shape", font_size=20, color=COLOR_STEPS)
        
        # Fix for Issue 55: Place step labels at E2, E4, E5 to avoid overlap with upper elements.
        self.place_at_grid(step1, "E2", scale_factor=0.8)
        self.place_at_grid(step2, "E4", scale_factor=0.8)
        self.place_at_grid(step3, "E5", scale_factor=0.8)
        
        self.play(Write(step1))
        self.play(Write(step2))
        self.play(Write(step3))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "The results will always follow a bell curve."
        self.lecture[3].set_color(COLOR_NORMAL)
        
        # Transform faint bell curve into a clear golden one
        clear_bell_plot = axes.plot(lambda x: 0.4 * np.exp(-x**2 / 2), color=COLOR_NORMAL, stroke_width=4)
        self.play(
            Transform(bell_curve_plot, clear_bell_plot),
            axes.animate.set_color(COLOR_NORMAL).set_stroke(opacity=1.0),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Statistics transforms chaos into clear, actionable order."
        self.lecture[4].set_color(COLOR_ORDER)
        
        order_text = Text("Order from Chaos", font_size=28, color=COLOR_ORDER)
        # Fix for Issue 56: Position order_text at grid A4 for centering.
        self.place_at_grid(order_text, "A4", scale_factor=1.1)
        
        # Glow effect for order_text
        glow = order_text.copy().set_stroke(width=10, opacity=0.3).set_color(COLOR_ORDER)
        
        self.play(Write(order_text), FadeIn(glow))
        self.wait(1.5)
        
        # Final visual: Clear Golden Bell Curve central to the universe
        final_axes = Axes(
            x_range=[-3, 3], y_range=[0, 0.5], 
            axis_config={"include_tip": False},
            tips=False
        ).set_color(COLOR_NORMAL)
        final_plot = final_axes.plot(lambda x: 0.4 * np.exp(-x**2 / 2), color=COLOR_NORMAL, stroke_width=6)
        final_bell_group = VGroup(final_axes, final_plot)
        self.place_in_area(final_bell_group, "B2", "E5", scale_factor=1.4)

        self.play(
            FadeOut(chaos_elements),
            FadeOut(bridge),
            FadeOut(step1), FadeOut(step2), FadeOut(step3),
            FadeOut(order_text), FadeOut(glow),
            Transform(bell_group, final_bell_group),
            run_time=2
        )
        self.wait(3)
