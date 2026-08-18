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
        self.setup_layout("The Central Limit Theorem Unveiled", [
            "The CLT is a fundamental statistical theorem.",
            "Sample means approach normal distribution as n grows.",
            "This holds regardless of the initial shape.",
            "It turns chaos into orderly patterns.",
            "Data becomes predictable at scale."
        ])
        
        # Load assets
        population_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/population.svg")
        curve_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/curve.svg")
        
        # Animation components
        distribution_area = VGroup(*[population_icon.copy() for _ in range(5)])
        self.place_in_area(distribution_area, "A1", "C2", scale_factor=0.4)
        distribution_area.set_color("#0088FF")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#0088FF")
        self.play(FadeIn(distribution_area))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        sample_mean = Dot(color="#FFFFFF", radius=0.1)
        self.place_at_grid(sample_mean, "B4", scale_factor=0.6)
        self.play(FadeIn(sample_mean))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF0000")
        normal_curve = FunctionGraph(lambda x: np.exp(-x**2), x_range=[-2, 2], color="#FF0000")
        self.place_in_area(normal_curve, "A4", "F6", scale_factor=0.6)
        self.play(Create(normal_curve))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        self.play(
            distribution_area.animate.set_opacity(0.3),
            sample_mean.animate.set_color("#FFFF00")
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        self.play(
            FadeOut(distribution_area),
            sample_mean.animate.move_to(normal_curve.get_center()),
            FadeIn(curve_icon.set_color("#FFFFFF").scale(0.8).move_to(normal_curve.get_center()))
        )
        self.wait(2)
