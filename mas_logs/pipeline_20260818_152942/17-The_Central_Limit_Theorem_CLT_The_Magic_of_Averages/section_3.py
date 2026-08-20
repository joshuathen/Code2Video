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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The CLT explains this magic.",
            "Sample means approach a normal distribution.",
            "This happens regardless of population shape.",
            "Increase sample size for better approximation.",
            "The bell curve always emerges."
        ]
        self.setup_layout("Core Concept: The Normal Approximation", lecture_lines)

        # Assets
        bell_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bell.svg")

        # Initial bar chart setup
        axes = Axes(x_range=[-4, 4, 1], y_range=[0, 1, 0.2], axis_config={"include_tip": False})
        chart = VGroup(*[Rectangle(height=0.1, width=0.5, color="#1E90FF", fill_opacity=1) for _ in range(5)])
        chart.arrange(RIGHT, buff=0.1)
        self.place_at_grid(chart, 'B3', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#1E90FF")
        self.play(FadeIn(chart))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#1E90FF")
        new_chart = VGroup(*[Rectangle(height=np.random.rand()*1.5, width=0.2, color="#1E90FF", fill_opacity=1) for _ in range(20)])
        new_chart.arrange(RIGHT, buff=0.05)
        self.place_at_grid(new_chart, 'D3', scale_factor=0.7)
        self.play(Transform(chart, new_chart))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        curve = FunctionGraph(lambda x: np.exp(-x**2), x_range=[-3, 3], color="#FFFF00")
        self.place_at_grid(curve, 'C5', scale_factor=0.9)
        self.play(Create(curve))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.play(FadeOut(chart))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF0000")
        visual_group = VGroup(curve, bell_icon)
        self.place_in_area(visual_group, 'B4', 'E6', scale_factor=0.85)
        self.play(FadeIn(bell_icon), Indicate(curve))
        self.wait(1)
