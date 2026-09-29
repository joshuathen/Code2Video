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
            "Algebraically, the dot product is sum of products.",
            "Geometrically, it relates to the cosine of theta.",
            "Think of it as projecting one vector.",
            "Visualize a flashlight casting a shadow.",
            "The length of this shadow is the product."
        ]
        self.setup_layout("Introduction: The Dot Product as a 'Shadow'", lecture_lines)
        
        # Mobjects for animations
        vec_u = Arrow(ORIGIN, RIGHT*1.5 + UP*0.8, color=BLUE)
        vec_v = Arrow(ORIGIN, RIGHT*2.0, color=YELLOW)
        label_u = MathTex("u", color=BLUE).scale(0.8)
        label_v = MathTex("v", color=YELLOW).scale(0.8)
        
        # Prepare for projection
        proj_line = DashedLine(vec_u.get_end(), vec_v.get_end(), color=GRAY)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        alg_formula = MathTex("u \\cdot v = \\sum u_i v_i", font_size=36)
        self.place_in_area(alg_formula, 'B2', 'B5', scale_factor=0.8)
        self.play(Write(alg_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        geo_formula = MathTex("u \\cdot v = |u||v| \\cos(\\theta)", font_size=36)
        self.place_in_area(geo_formula, 'C2', 'C5', scale_factor=0.8)
        self.play(ReplacementTransform(alg_formula.copy(), geo_formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(GREEN))
        self.place_at_grid(vec_u, 'D2', scale_factor=0.7)
        self.place_at_grid(vec_v, 'D4', scale_factor=0.7)
        label_u.next_to(vec_u.get_end(), UP)
        label_v.next_to(vec_v.get_end(), DOWN)
        self.add(vec_u, vec_v, label_u, label_v)
        self.play(Create(vec_u), Create(vec_v), Write(label_u), Write(label_v))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(PURPLE))
        flashlight = Circle(radius=0.3, color=PURPLE).shift(UP*1.5 + RIGHT*0.5)
        self.play(Create(flashlight))
        self.play(Create(proj_line))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(RED))
        projection_shadow = Line(ORIGIN, vec_v.get_end(), color=RED, stroke_width=6)
        self.place_in_area(projection_shadow, 'E2', 'F5', scale_factor=0.9)
        self.play(Create(projection_shadow))
        self.wait(2)
