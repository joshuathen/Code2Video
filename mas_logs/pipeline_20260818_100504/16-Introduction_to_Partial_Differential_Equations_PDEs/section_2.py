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
            "The heat equation is a fundamental PDE.",
            "Heat diffuses over a 2D surface.",
            "Time changes equal the Laplacian of space.",
            "Curvature drives heat redistribution across the plate.",
            "Temperature flows to reach thermal equilibrium."
        ]
        self.setup_layout("Core Concept: The Heat Equation", lecture_lines)
        
        # Define objects
        eq = MathTex(r"u_t = \alpha \nabla^2 u", font_size=48)
        plate_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/plate.svg"
        heatmap = SVGMobject(plate_asset).set_fill(color=BLUE, opacity=0.6)
        gaussian = Surface(
            lambda u, v: np.array([u, v, np.exp(-(u**2 + v**2))]),
            u_range=[-2, 2], v_range=[-2, 2],
            fill_opacity=0.8, color=GREEN
        )
        alpha_label = MathTex(r"\text{Diffusion Constant: } \alpha", font_size=32)

        # === Animation for Lecture Line 1 ===
        self.place_in_area(eq, 'C2', 'C5', scale_factor=0.8)
        self.play(FadeIn(eq))
        self.play(eq.animate.set_color("#FF5733"))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(heatmap, 'E3', scale_factor=1.0)
        self.play(FadeIn(heatmap))
        self.place_at_grid(gaussian, 'E5', scale_factor=0.4)
        self.play(Create(gaussian))
        self.play(heatmap.animate.set_fill(color="#33FF57"), gaussian.animate.set_color("#33FF57"))
        self.lecture[1].set_color("#33FF57")

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(alpha_label, 'C6', scale_factor=0.7)
        self.play(FadeIn(alpha_label))
        self.play(alpha_label.animate.set_color("#3357FF"))
        self.lecture[2].set_color("#3357FF")

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF5733")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(heatmap.animate.set_fill(color="#33FF57"))
        self.lecture[4].set_color("#33FF57")
        self.wait(1)
