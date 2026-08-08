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
        self.setup_layout("Prerequisite Review: The Normal Distribution", 
                          ["The Normal Distribution forms a symmetric bell curve.", 
                           "It is defined by its mean and variance.", 
                           "Nature often follows this smooth, predictable pattern."])
        
        # Create Normal Distribution curve
        axes = Axes(x_range=[-4, 4, 1], y_range=[0, 1, 0.2], axis_config={"include_tip": False}).scale(0.5)
        curve = axes.plot(lambda x: np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), color=BLUE)
        curve_group = VGroup(axes, curve)

        # === Animation for Lecture Line 1 ===
        # Fix: Centered curve_group
        self.place_in_area(curve_group, 'B1', 'E4', scale_factor=0.8)
        self.play(Create(curve_group), run_time=2)
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Fix: Added label μ with correct placement
        mu_label = MathTex(r"\mu", color=YELLOW)
        self.place_at_grid(mu_label, 'F3', scale_factor=0.7)
        self.play(Write(mu_label), run_time=1)
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Fix: Added highlight point with correct placement
        point_highlight = Dot(axes.c2p(1, 0.24), color=GREEN)
        self.place_at_grid(point_highlight, 'D4', scale_factor=0.6)
        self.play(FadeIn(point_highlight), run_time=1)
        self.lecture[2].set_color(GREEN)
        self.wait(2)
