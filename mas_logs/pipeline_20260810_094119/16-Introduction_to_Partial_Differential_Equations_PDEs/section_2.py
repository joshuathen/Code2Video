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
        self.setup_layout("The Heat Equation: Visualizing Diffusion", [
            "The Heat Equation models how diffusion spreads heat.",
            "It follows the formula: u_t = αu_xx.",
            "Visualize temperature evening out across a metal rod."
        ])

        # Asset
        rod = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg")

        # === Animation for Lecture Line 1 ===
        axes = Axes(x_range=[0, 6, 1], y_range=[0, 2, 0.5], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 1.5 * np.exp(-(x-3)**2), color=YELLOW)
        
        # Group rod and chart for synchronization
        content = VGroup(rod, axes, curve)
        
        self.place_in_area(content, 'C2', 'E5', scale_factor=0.55)
        
        self.play(Create(axes), Create(curve), FadeIn(rod))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        formula = MathTex("u_t = \\alpha u_{xx}", color="#00FF00")
        self.place_at_grid(formula, 'E5', scale_factor=0.7)
        self.play(Write(formula))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        smooth_curve = axes.plot(lambda x: 0.5 * np.exp(-(x-3)**2/2) + 0.5, color=BLUE)
        self.play(Transform(curve, smooth_curve), run_time=3)
        self.lecture[2].set_color(BLUE)
        self.wait(2)
