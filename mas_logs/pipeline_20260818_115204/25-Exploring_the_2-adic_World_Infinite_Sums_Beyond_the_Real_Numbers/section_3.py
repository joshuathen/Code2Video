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
            "Consider the geometric series 1 plus 2 plus 4.",
            "This sum diverges in the real numbers.",
            "In 2-adic arithmetic, it converges to minus 1.",
            "The accountant robot shows 1 plus 2S equals S.",
            "Solving confirms the sum equals minus 1."
        ]
        self.setup_layout("The Infinite Sum: 1 + 2 + 4 + ...", lecture_lines)
        
        # Load Assets
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        accountant = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/accountant.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        dots = VGroup(*[Dot(color=WHITE) for _ in range(4)])
        self.place_at_grid(robot, "B3", scale_factor=0.3)
        for i, dot in enumerate(dots):
            self.place_at_grid(dot, f"B{i+1}")
        self.play(FadeIn(robot), FadeIn(dots))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE),
                  self.lecture[1].animate.set_color("#FF5555"))
        self.play(FadeOut(dots))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE),
                  self.lecture[2].animate.set_color("#00FFFF"))
        sum_text = Text("S = -1", color="#00FFFF")
        self.place_at_grid(sum_text, "B4")
        self.play(Write(sum_text))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE),
                  self.lecture[3].animate.set_color("#AAFFAA"))
        eq = MathTex("1 + 2S = S").set_color("#AAFFAA")
        self.place_in_area(eq, "D4", "D6", scale_factor=0.8)
        self.play(Write(eq))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE),
                  self.lecture[4].animate.set_color("#FFFF00"))
        self.play(FadeOut(robot), FadeOut(eq))
        self.place_at_grid(accountant, "C3", scale_factor=0.5)
        self.play(FadeIn(accountant))
        self.wait(1)
