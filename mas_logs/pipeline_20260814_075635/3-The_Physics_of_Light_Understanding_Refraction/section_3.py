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
        self.setup_layout("Core Principle: Snell's Law", [
            "Snell's Law calculates light refraction angles.",
            "Incident rays meet the normal line.",
            "Light bends toward or away normal.",
            "Math predicts the path very accurately.",
            "Watch the light ray change direction."
        ])

        # Assets
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        
        # Objects
        formula = MathTex(r"n_1 \sin(\theta_1) = n_2 \sin(\theta_2)", color=WHITE)
        normal = DashedLine(start=UP*2, end=DOWN*2, color=LIGHT_GRAY)
        surface = Line(start=LEFT*3, end=RIGHT*3, color=BLUE)
        
        # Combined animation object for ray/prism
        ray_group = VGroup(laser, prism)
        ray_in = Line(start=LEFT*2+UP*1.5, end=ORIGIN, color=YELLOW)
        ray_out = Line(start=ORIGIN, end=RIGHT*1.5+DOWN*1, color=YELLOW)
        
        legend_label = Text("Snell's Law", font_size=20, color=WHITE)

        # Positioning
        self.place_in_area(formula, 'A4', 'B6', scale_factor=1.0)
        self.place_at_grid(prism, 'C3', scale_factor=0.5)
        self.place_in_area(VGroup(ray_in, ray_out, laser), 'D2', 'F5', scale_factor=0.9)
        self.place_at_grid(legend_label, 'C2', scale_factor=0.7)

        # Animations
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(formula))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(surface), FadeIn(normal), FadeIn(prism))
        self.lecture[1].set_color(LIGHT_GRAY)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(laser), Create(ray_in))
        self.lecture[2].set_color(BLUE)

        # === Animation for Lecture Line 4 ===
        self.play(Create(ray_out))
        self.lecture[3].set_color(GREEN)

        # === Animation for Lecture Line 5 ===
        self.play(ray_out.animate.rotate(-0.5, about_point=ORIGIN))
        self.lecture[4].set_color(RED)

        self.wait(2)
