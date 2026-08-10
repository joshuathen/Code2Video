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
            "We take repeated samples of size 30.",
            "Each sample mean is calculated and plotted.",
            "The chaos transforms into a bell curve.",
            "SampleMeanPlot grows with every iteration.",
            "This is the magic of the average."
        ]
        self.setup_layout("The Core Mechanism: The Sampling Process", lecture_lines)
        
        # Define objects
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/bar.svg]
        bar_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bar.svg")
        axes = Axes(x_range=[0, 10, 1], y_range=[0, 5, 1], axis_config={"include_numbers": False}).scale(0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        sampler_viz = bar_asset.copy()
        self.place_at_grid(sampler_viz, 'B2', scale_factor=0.6)
        self.play(FadeIn(sampler_viz), run_time=1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN)
        mean_label = Text("Mean", font_size=16, color=WHITE)
        mean_point = Dot(color="#2ECC71")
        self.place_at_grid(mean_point, 'E4', scale_factor=0.5)
        mean_label.next_to(mean_point, UP, buff=0.1)
        self.play(FadeIn(mean_point), FadeIn(mean_label), run_time=1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        self.place_in_area(axes, 'C2', 'F5', scale_factor=0.7)
        self.play(FadeOut(sampler_viz), FadeOut(mean_point), FadeOut(mean_label), Create(axes), run_time=1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(ORANGE)
        dots = VGroup(*[Dot(point=axes.c2p(np.random.normal(5, 1), np.random.uniform(0.1, 2)), color="#2ECC71", radius=0.03) for _ in range(50)])
        self.play(Create(dots), run_time=2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(BLUE)
        self.play(Indicate(axes), run_time=1)
