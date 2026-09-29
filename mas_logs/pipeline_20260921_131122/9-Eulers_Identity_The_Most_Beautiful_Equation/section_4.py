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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Grand Finale: e^(πi) = -1", [
            "Plug pi into the formula for x.",
            "cos pi is negative one; sin pi is zero.",
            "The equation becomes e to the pi i equals -1.",
            "This represents a half-turn around the origin.",
            "We have reached our destination: e to the pi i plus 1 equals 0."
        ])

        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")

        # Animation objects
        eq = MathTex("e^{ix} = \\cos(x) + i\\sin(x)", color=WHITE)
        self.place_in_area(eq, 'B2', 'C4', scale_factor=1.0)
        self.place_at_grid(compass, 'A5', scale_factor=0.5)
        self.play(FadeIn(eq), FadeIn(compass))
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        pi_sub = MathTex("e^{i\\pi} = \\cos(\\pi) + i\\sin(\\pi)", color=BLUE)
        self.place_in_area(pi_sub, 'B2', 'C4', scale_factor=1.0)
        self.play(Transform(eq, pi_sub))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        res = MathTex("\\cos(\\pi) = -1, \\quad \\sin(\\pi) = 0", color=GREEN)
        self.place_in_area(res, 'D2', 'E4', scale_factor=0.9)
        self.play(Write(res))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        final_eq = MathTex("e^{i\\pi} = -1", color=RED)
        self.place_at_grid(final_eq, 'B2', scale_factor=1.5)
        self.place_at_grid(protractor, 'F5', scale_factor=0.5)
        self.play(FadeOut(eq), FadeOut(res), FadeIn(final_eq), FadeIn(protractor))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        ultimate = MathTex("e^{i\\pi} + 1 = 0", color=GOLD)
        self.place_in_area(ultimate, 'B5', 'D6', scale_factor=1.2)
        self.play(Transform(final_eq, ultimate))
        self.wait(2)
