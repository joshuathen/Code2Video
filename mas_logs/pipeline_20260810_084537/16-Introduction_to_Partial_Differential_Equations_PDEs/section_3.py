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
        lecture_lines = ["The heat equation models temperature diffusion.", "Temperature change relates to spatial curvature.", "Heat flows from hot to cold regions."]
        self.setup_layout("The Heat Equation: A Core Example", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        eq = MathTex(r"u_t = \alpha u_{xx}", font_size=40)
        self.place_at_grid(eq, 'B4', scale_factor=1.2)
        self.play(Write(eq))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Represent curvature with a simple function
        axes = Axes(x_range=[-2, 2], y_range=[-1, 2], x_length=3, y_length=2).set_color(GRAY)
        curve = axes.plot(lambda x: 1 - x**2, color=RED)
        graph_group = VGroup(axes, curve)
        self.place_in_area(graph_group, 'C3', 'D6', scale_factor=0.5)
        self.play(Create(graph_group))
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Heatmap / Diffusion
        # Use asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg]
        rod = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg")
        rod.set_fill(color=RED, opacity=0.8)
        self.place_at_grid(rod, 'E4', scale_factor=0.9)
        self.play(FadeIn(rod))
        self.play(rod.animate.set_fill(color=BLUE, opacity=0.8), run_time=2)
        self.lecture[2].set_color(YELLOW)
        self.wait(1)
