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
            "We define superposition as a linear combination.",
            "The state vector is ψ equals alpha |0> plus beta |1>.",
            "Alpha and beta are complex probability amplitudes.",
            "The sum of squares must equal one.",
            "Visually, this maps to the Bloch sphere surface."
        ]
        self.setup_layout("Defining Superposition", lecture_lines)
        
        # Asset path
        sphere_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        psi_eq = MathTex(r"\psi = \alpha |0\rangle + \beta |1\rangle")
        self.place_at_grid(psi_eq, 'B2', scale_factor=1.0)
        self.play(Write(psi_eq))
        
        # Add sphere asset
        sphere_icon = SVGMobject(sphere_asset)
        self.place_at_grid(sphere_icon, 'B5', scale_factor=0.3)
        self.play(FadeIn(sphere_icon))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        alpha_text = MathTex(r"\alpha")
        beta_text = MathTex(r"\beta")
        alpha_text.set_color(ORANGE)
        beta_text.set_color(PURPLE)
        self.place_at_grid(alpha_text, 'C2', scale_factor=0.8)
        self.place_at_grid(beta_text, 'C4', scale_factor=0.8)
        self.play(FadeIn(alpha_text), FadeIn(beta_text))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(RED)
        norm_eq = MathTex(r"|\alpha|^2 + |\beta|^2 = 1")
        self.place_at_grid(norm_eq, 'E2', scale_factor=0.8)
        self.play(Write(norm_eq))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(TEAL)
        # Using SVGMobject sphere per asset requirement
        sphere_bloch = SVGMobject(sphere_asset)
        self.place_at_grid(sphere_bloch, 'C6', scale_factor=0.6)
        self.play(Create(sphere_bloch))
        self.wait(2)
