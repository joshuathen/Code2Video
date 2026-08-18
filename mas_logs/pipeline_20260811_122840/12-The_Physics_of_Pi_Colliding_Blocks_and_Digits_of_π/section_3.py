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
        self.setup_layout("The Mass Ratio Connection", [
            "Collision counts depend on mass ratios.",
            "Mass ratios follow powers of one hundred.",
            "Bounces reveal consecutive digits of pi.",
            "Higher ratios yield more pi digits.",
            "This links mechanics to number theory."
        ])

        # Objects
        m1_label = Text("m₁", color=WHITE)
        m2_label = Text("m₂", color=WHITE)
        ratio_text = MathTex(r"M/m = 100^n", color="#FFFFFF")
        circles = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circles.svg")
        
        # Setup visual elements
        # Repositioned per Issue 26
        self.place_at_grid(m1_label, "B2", scale_factor=0.9)
        self.place_at_grid(m2_label, "B5", scale_factor=0.9)
        # Repositioned per Issue 27
        self.place_in_area(ratio_text, "C2", "C5", scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Write(m1_label), Write(m2_label), Write(ratio_text))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF6600"))
        # Integrates Asset per Issue 18
        self.place_in_area(circles, "D2", "E5", scale_factor=0.7)
        self.play(FadeIn(circles), ratio_text.animate.set_color("#FF6600"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        pi_approx = MathTex(r"\\approx 3.14159...", color="#00FF00")
        # Repositioned per Issue 28
        self.place_in_area(pi_approx, "D3", "D4", scale_factor=1.1)
        self.play(Write(pi_approx))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        self.wait(2)
