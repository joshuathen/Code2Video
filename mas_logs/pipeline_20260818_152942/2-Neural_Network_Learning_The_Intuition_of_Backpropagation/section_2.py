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
        self.setup_layout("Prerequisite: The Gradient as a 'Directional Slope'", [
            "Visualize loss as a mountainous landscape.",
            "The goal is reaching the lowest valley.",
            "'Gradient' shows the direction of steepest ascent."
        ])
        
        # Surface setup
        axes = ThreeDAxes(x_range=[-3, 3], y_range=[-3, 3], z_range=[0, 3], axis_config={"include_tip": False})
        surface = axes.plot_surface(
            lambda u, v: 0.2 * (u**2 + v**2),
            u_range=[-2.5, 2.5],
            v_range=[-2.5, 2.5],
            resolution=(20, 20),
            color="#808080"
        )
        
        mountain_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg", color="#808080")
        
        # Combined plot group
        landscape = VGroup(axes, surface, mountain_icon)
        self.place_in_area(landscape, 'B2', 'E5', scale_factor=0.55)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(landscape))
        self.lecture[0].set_color("#808080")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        white_dot = Dot(color="#FFFFFF")
        self.place_at_grid(white_dot, 'C3', scale_factor=0.4)
        self.add(white_dot)
        
        # Move dot
        self.play(white_dot.animate.move_to(axes.c2p(0, 0, 0)), run_time=2)
        self.lecture[1].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        gradient_arrow = Arrow(start=axes.c2p(0, 0, 0), end=axes.c2p(1, 1, 0.4), color="#FFFF00")
        self.place_at_grid(gradient_arrow, 'C4', scale_factor=0.7)
        self.play(Create(gradient_arrow))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
