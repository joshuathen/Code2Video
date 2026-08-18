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
        lecture_lines = [
            "Snell's Law explains light bending.",
            "The Normal line is perpendicular.",
            "Light slows and bends toward it.",
            "Entering glass, the angle closes.",
            "It bends away when exiting."
        ]
        self.setup_layout("The Physics of Refraction (Snell’s Law)", lecture_lines)

        # Assets
        glass_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg")
        
        # Define graphical elements
        normal = DashedLine(start=[0, 2, 0], end=[0, -2, 0], color=GRAY)
        
        incident_ray = Line(start=[-2, 1.5, 0], end=[0, 0, 0], color="#FFD700")
        theta1 = MathTex(r"\\theta_1", color="#FFD700")
        
        refracted_ray = Line(start=[0, 0, 0], end=[1.5, -1.2, 0], color="#00FF00")
        theta2 = MathTex(r"\\theta_2", color="#00FF00")
        
        snell_eq = MathTex(r"n_1 \\sin(\\theta_1) = n_2 \\sin(\\theta_2)", font_size=36)
        
        # Grid positioning
        self.place_at_grid(glass_asset, 'C4', scale_factor=0.6)
        self.place_at_grid(normal, 'C4', scale_factor=0.8)
        self.place_at_grid(incident_ray, 'C4', scale_factor=0.8)
        self.place_at_grid(refracted_ray, 'C4', scale_factor=0.8)
        
        self.place_at_grid(theta1, 'B3', scale_factor=0.8)
        self.place_at_grid(theta2, 'E3', scale_factor=0.8)
        self.place_in_area(snell_eq, 'C4', 'D6', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(Create(incident_ray), Write(theta1))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.play(Create(normal), FadeIn(glass_asset))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.play(Create(refracted_ray), Write(theta2))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        self.play(Write(snell_eq))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFFFF"))
        self.play(Indicate(snell_eq))
        self.wait(1)
