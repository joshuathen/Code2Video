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
        lecture_lines = [
            "Partial derivatives measure rates in specific directions.",
            "Imagine a hiker traversing a mountain surface.",
            "Slope changes differently along East-West versus North-South paths.",
            "We hold one coordinate constant while varying another.",
            "This reveals the rate of change in that direction."
        ]
        self.setup_layout("Core Mechanism: Partial Derivatives", lecture_lines)
        
        # Assets
        hiker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hiker.svg")
        mountain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_at_grid(hiker, "B1", scale_factor=0.3)
        self.play(FadeIn(hiker))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF0000")
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-1, 2], axis_config={"include_tip": False})
        surface = axes.plot_surface(lambda u, v: 0.5 * (np.sin(u) + np.cos(v)), u_range=[-2, 2], v_range=[-2, 2]).set_color(RED).set_opacity(0.6)
        axes_surface_group = VGroup(axes, surface, mountain.copy().scale(0.2))
        self.place_in_area(axes_surface_group, "B3", "E6", scale_factor=0.5)
        self.play(Create(axes), Create(surface), FadeIn(mountain.scale(0.1).move_to(self.grid["A3"])))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        slice_x = Surface(lambda u, v: axes.c2p(u, 0, 0.5 * (np.sin(u) + 1)), u_range=[-2, 2], v_range=[-1, 1]).set_color(GREEN).set_opacity(0.8)
        self.play(Create(slice_x))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#0000FF")
        slice_y = Surface(lambda u, v: axes.c2p(0, v, 0.5 * (0 + np.cos(v))), u_range=[-1, 1], v_range=[-2, 2]).set_color(BLUE).set_opacity(0.8)
        self.play(Create(slice_y))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        text_deriv = Text("∂f/∂x and ∂f/∂y", font_size=20, color=YELLOW)
        self.place_at_grid(text_deriv, "D4", scale_factor=1.0)
        self.place_at_grid(mountain.copy(), "E4", scale_factor=0.1)
        self.play(Write(text_deriv))
        self.wait(2)
