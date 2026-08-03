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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Convolution blends two sets of information together.",
            "It modifies a signal using a specific filter.",
            "Think of it as a weighted moving average.",
            "We apply weights to the input data over time.",
            "This process creates a refined, filtered version of reality."
        ]
        self.setup_layout("The Big Picture: What is Convolution?", lecture_lines)

        # Colors
        COLOR_SIGNAL = "#FFD700"  # Gold
        COLOR_FILTER = "#FF69B4"  # HotPink
        COLOR_RESULT = "#90EE90"  # LightGreen
        COLOR_WEIGHT = "#FFA500"  # Orange
        COLOR_TITLE = "#ADD8E6"   # LightBlue

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_TITLE))
        
        signal_circle = Circle(radius=0.6, color=COLOR_SIGNAL, fill_opacity=0.5)
        signal_label = Text("Signal", font_size=20, color=COLOR_SIGNAL)
        signal_group = VGroup(signal_circle, signal_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(signal_group, "B2")

        filter_circle = Circle(radius=0.6, color=COLOR_FILTER, fill_opacity=0.5)
        filter_label = Text("Filter", font_size=20, color=COLOR_FILTER)
        filter_group = VGroup(filter_circle, filter_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(filter_group, "B5")

        self.play(FadeIn(signal_group), FadeIn(filter_group))
        self.wait(1)

        result_circle = Circle(radius=0.8, color=COLOR_RESULT, fill_opacity=0.7)
        result_label = Text("Result", font_size=24, color=COLOR_RESULT)
        result_group = VGroup(result_circle, result_label).arrange(DOWN, buff=0.1)
        # Resolved Issue 35: Updated area for result_group
        self.place_in_area(result_group, "D3", "E4")

        # Blend circles into result
        self.play(
            signal_group.animate.move_to(self.grid["C3"]),
            filter_group.animate.move_to(self.grid["C4"]),
            run_time=1.5
        )
        self.play(
            ReplacementTransform(VGroup(signal_group, filter_group), result_group)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_FILTER)
        )

        # Create discrete bar charts
        def create_bar_chart(values, color, label_text):
            bars = VGroup(*[Rectangle(width=0.2, height=v*0.3, fill_color=color, fill_opacity=0.8, stroke_width=1) for v in values])
            bars.arrange(RIGHT, aligned_edge=DOWN, buff=0.1)
            label = Text(label_text, font_size=18, color=color)
            return VGroup(bars, label).arrange(DOWN, buff=0.2)

        signal_bars = create_bar_chart([2, 3, 1, 4, 2], COLOR_SIGNAL, "Signal")
        filter_bars = create_bar_chart([1, 2, 1], COLOR_FILTER, "Filter")
        
        # Resolved Issue 37: Increased horizontal gap between signal and filter bars
        self.place_in_area(signal_bars, "B1", "C2")
        self.place_in_area(filter_bars, "B5", "C6")

        self.play(
            ReplacementTransform(result_group, VGroup(signal_bars, filter_bars))
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_WEIGHT)
        )
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_WEIGHT)
        )
        
        weight_label = Text("Weight", font_size=20, color=COLOR_WEIGHT)
        self.place_at_grid(weight_label, "A5")
        
        arrow = Arrow(start=self.grid["A5"], end=self.grid["B5"], color=COLOR_WEIGHT, buff=0.1)
        
        self.play(FadeIn(weight_label), GrowArrow(arrow))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_RESULT)
        )

        result_bars = create_bar_chart([2, 7, 9, 10, 10, 8, 2], COLOR_RESULT, "Weighted Moving Average")
        # Resolved Issue 36: Fixed placement and scale for result_bars
        self.place_in_area(result_bars, "E2", "F5", scale_factor=0.8)

        self.play(
            FadeIn(result_bars),
            FadeOut(weight_label),
            FadeOut(arrow)
        )
        self.wait(2)
