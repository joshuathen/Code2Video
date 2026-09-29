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
            "Primes bridge the discrete to the continuous.",
            "The Euler Product Formula connects primes to Pi.",
            "Reciprocals of prime squares converge.",
            "Infinite products reveal circular geometry.",
            "Robot arm draws a perfect circle."
        ]
        self.setup_layout("Approximating Pi via Prime Logic", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg", color=WHITE)
        self.place_at_grid(robot, 'A3', scale_factor=0.3)
        self.play(DrawBorderThenFill(robot))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        formula = MathTex(r"\zeta(s) = \prod_{p} \frac{1}{1-p^{-s}}", font_size=36)
        self.place_in_area(formula, 'B3', 'B5', scale_factor=0.8)
        self.play(Write(formula))
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        sum_text = MathTex(r"\sum \frac{1}{p^2} \approx 0.452", font_size=32)
        self.place_in_area(sum_text, 'C3', 'C5', scale_factor=0.8)
        self.play(FadeIn(sum_text))
        self.lecture[2].set_color("#FF4500")

        # === Animation for Lecture Line 4 ===
        circle = Circle(radius=1.0, color=WHITE)
        self.place_at_grid(circle, 'D4', scale_factor=0.7)
        self.play(Create(circle))
        self.lecture[3].set_color("#FFFFFF")

        # === Animation for Lecture Line 5 ===
        arm = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arm.svg", color=BLUE)
        self.place_at_grid(arm, 'D6', scale_factor=0.6)
        self.play(Rotate(arm, angle=2*PI, about_point=arm.get_center()))
        self.lecture[4].set_color("#FF4500")
