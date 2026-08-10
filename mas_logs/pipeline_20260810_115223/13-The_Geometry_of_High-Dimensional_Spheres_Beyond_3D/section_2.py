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
        lecture_lines_text = [
            "Unit spheres shrink as dimensions increase.",
            "The volume tends toward zero.",
            "We use the Gamma function for calculation.",
            "Imagine a unit sphere in a unit cube.",
            "Volume migrates into the corners."
        ]
        self.setup_layout("The Counter-Intuitive Volume Paradox", lecture_lines_text)

        # Assets
        sphere_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        cube_img = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        
        # Mobjects
        corners = VGroup(*[Square(side_length=0.4, color="#FF4500", fill_opacity=0.8) for _ in range(4)])
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#1E90FF"))
        self.place_at_grid(sphere_img, 'C2', scale_factor=0.6)
        self.play(FadeIn(sphere_img))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.play(sphere_img.animate.scale(0.5))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#32CD32"))
        gamma_formula = MathTex(r"V_n(R) = \frac{\pi^{n/2}}{\Gamma(n/2+1)}R^n", color=WHITE)
        self.place_at_grid(gamma_formula, 'F3', scale_factor=0.7)
        self.play(Write(gamma_formula))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF4500"))
        self.place_at_grid(cube_img, 'C4', scale_factor=0.6)
        self.play(FadeIn(cube_img))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF4500"))
        
        self.place_at_grid(corners[0], 'A1', scale_factor=0.4)
        self.place_at_grid(corners[1], 'A5', scale_factor=0.4)
        self.place_at_grid(corners[2], 'E1', scale_factor=0.4)
        self.place_at_grid(corners[3], 'E5', scale_factor=0.4)
        
        self.play(FadeIn(corners))
        self.play(sphere_img.animate.scale(0.3))
        self.wait(2)
