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
        lecture_lines = [
            "Circles in 2D define spheres in 3D.",
            "Generalizing, spheres in n-dimensions live in R^n.",
            "Points distance r from origin: sum x_i^2 = r^2.",
            "[Asset: 2D_Circle] morphs into [Asset: 3D_Sphere].",
            "Projection represents [Asset: 4D_Hypersphere]."
        ]
        self.setup_layout("Introduction: The Curse of Dimensionality", lecture_lines)
        
        # Load assets
        hypercube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        sphere_shell = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        
        inner_shell = Circle(color="#FF4500", fill_opacity=0.6)
        dots = VGroup(*[Dot(color="#00CED1", radius=0.05) for _ in range(8)])
        arrows = VGroup(*[Arrow(start=ORIGIN, end=RIGHT*0.5, color="#FFD700", buff=0) for _ in range(4)])
        empty_text = Text("Empty Space", color="#FFFFFF", font_size=24)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        hypercube.set_color("#FFFFFF")
        self.place_in_area(hypercube, 'C3', 'F5', scale_factor=1.0)
        self.play(Create(hypercube), FadeIn(inner_shell.scale(0.3).move_to(hypercube.get_center())))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00CED1")
        for i, dot in enumerate(dots):
            self.place_at_grid(dot, ['B2', 'B5', 'E2', 'E5', 'C3', 'C4', 'D3', 'D4'][i], scale_factor=1.0)
        self.play(FadeIn(dots))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        self.play(FadeOut(inner_shell))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFD700")
        for i, arrow in enumerate(arrows):
            arrow.move_to(hypercube.get_center())
            arrow.rotate(i * PI/2)
        self.play(Create(arrows))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        self.place_in_area(sphere_shell, 'C3', 'C3', scale_factor=0.3)
        self.place_at_grid(empty_text, 'C5', scale_factor=0.8)
        self.play(FadeIn(sphere_shell), Write(empty_text))
        self.wait(2)
