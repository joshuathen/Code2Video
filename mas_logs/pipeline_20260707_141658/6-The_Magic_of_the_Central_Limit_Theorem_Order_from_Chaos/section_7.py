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

class Section7Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Real-World Application: Quality Control",
            [
                "We monitor candy bars on a factory line.",
                "Sample means must fall within safe bell limits.",
                "This guarantees consistent quality for every batch."
            ]
        )

        # Colors
        ORANGE_C = "#FFA500"
        BLUE_C = "#0000FF"
        GREEN_C = "#00FF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(ORANGE_C)
        
        # Create a row of orange rectangles (candy bars)
        candy_bars = VGroup(*[
            Rectangle(width=0.4, height=0.2, color=ORANGE_C, fill_opacity=0.8, fill_color=ORANGE_C)
            for _ in range(6)
        ]).arrange(RIGHT, buff=0.2)
        # Fix Issue 52: Move to D1-D6
        self.place_in_area(candy_bars, "D1", "D6")
        
        # Scanner line
        scanner_line = Line(
            start=candy_bars.get_left() + UP * 0.4,
            end=candy_bars.get_left() + DOWN * 0.4,
            color=BLUE_C,
            stroke_width=6
        )
        
        self.play(Create(candy_bars))
        self.add(scanner_line)
        self.play(scanner_line.animate.move_to(candy_bars.get_right()), run_time=2)
        self.play(FadeOut(scanner_line))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN_C)
        
        # Bell Curve
        curve_axes = Axes(
            x_range=[-3, 3], y_range=[0, 0.5],
            axis_config={"include_tip": False},
            x_length=4, y_length=2
        )
        
        def normal_pdf(x):
            return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
        
        bell_curve = curve_axes.plot(normal_pdf, color=WHITE)
        shaded_region = curve_axes.get_area(bell_curve, x_range=[-1, 1], color=GREEN_C, opacity=0.3)
        
        # Group axes and plots
        curve_group = VGroup(curve_axes, bell_curve, shaded_region)
        # Fix Issue 51: Move to A1-C6
        self.place_in_area(curve_group, "A1", "C6")
        
        label_safe = Text("Safe Limits", font_size=16, color=GREEN_C)
        # Fix Issue 51: Move to C3-C4
        self.place_in_area(label_safe, "C3", "C4", scale_factor=0.8)

        self.play(Create(bell_curve))
        self.play(FadeIn(shaded_region), FadeIn(label_safe))
        
        # A single dot (batch average) falls into the limits
        dot = Dot(color=BLUE_C)
        # Position it inside the bell curve center hump
        dot_pos = curve_axes.c2p(0, 0.2)
        dot.move_to(dot_pos)
        self.play(FadeIn(dot, shift=DOWN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN_C)
        
        quality_text = Text("Quality Guaranteed", font_size=24, color=GREEN_C)
        # Fix Issue 53: Move to E1-E6
        self.place_in_area(quality_text, "E1", "E6", scale_factor=1.0)
        
        # Subtle glow
        glow = quality_text.copy().set_style(stroke_width=8, stroke_color=GREEN_C, stroke_opacity=0.3)
        
        self.play(FadeIn(glow), Write(quality_text))
        self.wait(2)
