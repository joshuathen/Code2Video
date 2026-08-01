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
        # Fetching storyboard data
        title_text = "Introduction: The Hidden Pattern"
        lecture_lines = [
            "Some data distributions look messy and chaotic.",
            "This randomness creates a sense of total chaos.",
            "A hidden order emerges when we group data.",
            "The peak represents the most common sample mean.",
            "This phenomenon is the Central Limit Theorem."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors for matching animation elements
        COLOR_HIST = "#FF0000"
        COLOR_CHAOS = "#FFFFFF"
        COLOR_CURVE = "#0000FF"
        COLOR_PEAK = "#FFFF00"
        COLOR_CLT = "#FFFFFF"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_HIST))
        
        # Create a chaotic, bimodal histogram
        bar_heights = [1.2, 2.5, 1.0, 0.4, 0.6, 2.2, 2.8, 1.4]
        histogram = VGroup(*[
            Rectangle(
                width=0.4, 
                height=h, 
                fill_opacity=0.8, 
                fill_color=COLOR_HIST, 
                stroke_color=WHITE, 
                stroke_width=1
            )
            for h in bar_heights
        ]).arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        
        # Applied fix for Issue 35: scale_factor=0.8 for padding
        self.place_in_area(histogram, "B1", "E6", scale_factor=0.8)
        self.play(Create(histogram))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_CHAOS)
        )
        
        chaos_text = Text("CHAOS", font_size=36, weight=BOLD, color=COLOR_CHAOS)
        # Applied fix for Issue 34: Narrower area A2-A5 and scale_factor=0.8
        self.place_in_area(chaos_text, "A2", "A5", scale_factor=0.8)
        
        self.play(Write(chaos_text))
        self.play(Indicate(chaos_text, color=COLOR_HIST))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_CURVE)
        )
        
        # Create a smooth blue Bell Curve (Normal Distribution approximation)
        # Note: Using Axes locally to define the plot curve geometry
        axes = Axes(
            x_range=[-3, 3], y_range=[0, 1],
            x_length=5, y_length=3,
            axis_config={"include_tip": False}
        )
        
        curve = axes.plot(lambda x: np.exp(-x**2), color=COLOR_CURVE, x_range=[-2.5, 2.5])
        curve_group = VGroup(curve)
        # Maintain consistent scaling and area with the histogram
        self.place_in_area(curve_group, "B1", "E6", scale_factor=0.8)
        
        self.play(
            FadeOut(chaos_text),
            ReplacementTransform(histogram, curve_group)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_PEAK)
        )
        
        # Highlight the peak of the Bell Curve with a yellow vertical line
        peak_top = curve.get_top()
        # Find the bottom of the curve bounds
        peak_bottom_y = curve_group.get_bottom()[1]
        peak_line = Line(
            start=peak_top,
            end=[peak_top[0], peak_bottom_y, 0],
            color=COLOR_PEAK,
            stroke_width=5
        )
        
        self.play(Create(peak_line))
        self.play(Flash(peak_line, color=COLOR_PEAK))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_CLT)
        )
        
        clt_text = Text("Central Limit Theorem", font_size=32, color=COLOR_CLT)
        # Applied fix for Issue 36: scale_factor=0.8 for the bottom label
        self.place_in_area(clt_text, "F1", "F6", scale_factor=0.8)
        
        self.play(FadeIn(clt_text))
        self.wait(2)
        
        # Reset color and finish
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(2)
