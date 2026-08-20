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
        self.setup_layout("Application: The Law of Large Numbers in Action", [
            "CLT enables precise hypothesis testing and confidence intervals.",
            "We predict outcomes without knowing the full population.",
            "Predicting future statistical behavior becomes reliable."
        ])

        # --- Animation Objects ---
        # 1. Simulation of coin flips
        coin_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color="#00FFFF")
        coin_dots = VGroup(*[coin_icon.copy().scale(0.1) for _ in range(10)])
        self.place_in_area(coin_dots, "D1", "F3", scale_factor=0.6)

        # 2. Running average
        axes = Axes(x_range=[0, 50, 10], y_range=[0, 1, 0.5], axis_config={"include_numbers": False}).scale(0.5)
        graph = Line(start=axes.c2p(0, 0.5), end=axes.c2p(50, 0.5), color="#00FF00")
        plot_group = VGroup(axes, graph)
        self.place_at_grid(plot_group, "E5", scale_factor=0.7)

        # 3. Confidence interval
        ci = Polygon(axes.c2p(5, 0.4), axes.c2p(5, 0.6), axes.c2p(45, 0.55), axes.c2p(45, 0.45), fill_opacity=0.3, color="#FFFF00")
        ci_icon = coin_icon.copy().scale(0.1).set_color("#FFFF00")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"), Create(coin_dots), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"), Create(plot_group), run_time=2)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"), FadeIn(ci), FadeIn(ci_icon.next_to(ci, UP)), run_time=2)
        self.wait(2)
