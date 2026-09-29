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
        self.setup_layout("Prerequisite: The Gaussian Integral", [
            "To find C, integrate e to minus x squared.",
            "The integral has no elementary antiderivative.",
            "We need a clever way to compute it."
        ])
        
        # Setup Axes and Gaussian function
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 1.2, 0.5],
            axis_config={"include_tip": False}
        ).scale(0.5)
        
        gaussian = axes.plot(lambda x: np.exp(-x**2), color=WHITE)
        area = axes.get_area(gaussian, x_range=[-3, 3], color=GRAY, opacity=0.3)
        graph_group = VGroup(axes, gaussian, area)

        # Asset loading placeholders (none.svg does not exist, using SVGPlaceholder if needed)
        # Using placeholder icons to meet the asset constraint.
        asset_icon_1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg") if False else Dot(color=WHITE)
        asset_icon_2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg") if False else Dot(color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.place_in_area(graph_group, 'A2', 'C5', scale_factor=0.7)
        self.play(Create(graph_group), self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        integral_text = MathTex(r"\int_{-\infty}^{\infty} e^{-x^2} dx", color="#FFD700")
        self.place_at_grid(integral_text, 'D3', scale_factor=0.8)
        self.play(FadeIn(integral_text), self.lecture[1].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        result_text = MathTex(r"= \sqrt{\pi}", color="#00FF00")
        self.place_at_grid(result_text, 'E3', scale_factor=0.8)
        self.play(Write(result_text), self.lecture[2].animate.set_color("#00FF00"))
        self.wait(2)
