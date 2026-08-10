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
            "Kolmogorov theorized energy cascades in 1941.",
            "Energy dissipation rate (ε) drives the process.",
            "Nested eddies follow self-similar fractal patterns.",
            "Viscosity converts kinetic energy into microscopic heat.",
            "Energy spectrum follows a -5/3 power law."
        ]
        self.setup_layout("The Kolmogorov Cascade: Mathematical Scaling", lecture_lines)
        
        # Mobjects
        eddy_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eddy.svg", color=BLUE)
        heat_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/heat.svg", color=RED)
        
        cascade_group = VGroup(*[eddy_icon.copy().scale(0.5 - i*0.05).shift(RIGHT*i*0.2) for i in range(5)])
        energy_source = Dot(color="#00BFFF", radius=0.2)
        dissipation_dot = heat_icon
        spectrum_label = MathTex(r"k^{-5/3}", font_size=40)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFD700")
        self.place_in_area(cascade_group, 'C4', 'E5', scale_factor=0.8)
        self.play(Create(cascade_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00BFFF")
        self.place_at_grid(energy_source, 'A5', scale_factor=1.0)
        self.play(FadeIn(energy_source))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#32CD32")
        self.play(cascade_group.animate.arrange(RIGHT, buff=-0.3).scale(0.8))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF0000")
        self.place_at_grid(dissipation_dot, 'E6', scale_factor=0.5)
        self.play(FadeIn(dissipation_dot))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#9370DB")
        self.place_in_area(spectrum_label, 'F2', 'F5', scale_factor=1.2)
        self.play(Write(spectrum_label))
        
        self.wait(2)
