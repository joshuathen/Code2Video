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
            "Summing independent Gaussians results in another Gaussian.",
            "New mean is the sum of original means.",
            "New variance is the sum of original variances."
        ]
        self.setup_layout("The Addition Principle (Intuitive Phase)", lecture_lines)
        
        # Elements (Using Assets)
        scales = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scales.svg")
        containers = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/containers.svg")
        weights = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/weights.svg")
        
        label1 = Text("N(μ₁, σ₁²)", font_size=20, color="#FF00FF")
        label2 = Text("N(μ₂, σ₂²)", font_size=20, color="#00FFFF")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        # Using scales.svg for the two disjoint sets
        self.place_at_grid(scales, "C2", scale_factor=0.7)
        scales.set_color("#FF00FF")
        self.place_at_grid(label1, "D2", scale_factor=0.7)
        # Assuming we need another instance for the second set or split the icon
        scales2 = scales.copy().set_color("#00FFFF")
        self.place_at_grid(scales2, "C5", scale_factor=0.7)
        self.place_at_grid(label2, "D5", scale_factor=0.7)
        self.play(FadeIn(scales), Write(label1), FadeIn(scales2), Write(label2))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        self.place_in_area(containers, "C3", "D4", scale_factor=0.6)
        containers.set_color("#FFFFFF")
        self.play(Create(containers))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        self.place_at_grid(weights, "E3", scale_factor=0.5)
        self.play(FadeOut(scales), FadeOut(label1), FadeOut(scales2), FadeOut(label2), FadeOut(containers))
        self.play(FadeIn(weights))
        self.wait(1)
