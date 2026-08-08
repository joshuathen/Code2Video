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
        self.setup_layout("The Problem: Randomness and Chaos", [
            "Real-world data is often messy and non-normal.",
            "Individual events are unpredictable and chaotic.",
            "How do we predict averages from chaotic sources?"
        ])
        
        # Create Bar Chart for uniform dice distribution
        bar_chart = BarChart(
            values=[10, 10, 10, 10, 10, 10],
            bar_names=["1", "2", "3", "4", "5", "6"],
            y_range=[0, 12, 2],
            y_axis_config={"include_tip": False},
        ).set_color_by_gradient(BLUE, GREEN)

        # Fixed layout based on Issue 24, 25, 26
        self.place_in_area(bar_chart, 'A2', 'E5', scale_factor=0.65)
        
        header = Text("Uniform Distribution", font_size=20, color=BLUE)
        self.place_at_grid(header, 'A3', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(bar_chart), Write(header))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Scattered red data points (Issue 25)
        data_points = VGroup(*[Dot(radius=0.08, color=RED) for _ in range(10)])
        self.place_in_area(data_points, 'A5', 'F6', scale_factor=0.5)
        
        self.play(FadeIn(data_points))
        self.wait(1)
        self.play(FadeOut(data_points))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(2)
