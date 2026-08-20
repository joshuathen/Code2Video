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
        self.setup_layout("The Intuition of 'Nearness'", [
            "Euclidean distance measures gaps on a number line.",
            "P-adic distance counts how often 2 divides.",
            "High powers of 2 make numbers feel small.",
            "1024 is very close to 0 in 2-adics.",
            "Nearness depends entirely on our metric choice."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        line = NumberLine(x_range=[-1, 5, 1], length=5, include_numbers=True, font_size=24)
        point_a = Dot(color=BLUE).move_to(line.n2p(0))
        point_b = Dot(color=RED).move_to(line.n2p(2))
        group = VGroup(ruler, line, point_a, point_b)
        # Applying fix for VideoCritic issue 20/35
        self.place_in_area(group, 'B4', 'F6', scale_factor=0.6)
        self.play(FadeIn(ruler), Create(line), FadeIn(point_a), FadeIn(point_b))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        calc = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        formula = MathTex(r"|x|_2 = 2^{-v_2(x)}").set_color(GREEN)
        # Applying fix for VideoCritic issue 21/36
        self.place_at_grid(formula, 'B2', scale_factor=0.9)
        self.place_at_grid(calc, 'B3', scale_factor=0.5)
        self.play(Write(formula), FadeIn(calc))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        desc = Text("Power of 2 grows -> distance shrinks", font_size=20).set_color(BLUE)
        # Applying fix for VideoCritic issue 22/37
        self.place_at_grid(desc, 'D2', scale_factor=0.7)
        self.play(FadeIn(desc))

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        zero = Dot(color=BLUE).move_to(self.grid["C3"])
        large = Dot(color=RED).move_to(self.grid["C4"])
        label_1024 = MathTex("1024").next_to(large, RIGHT, buff=0.1).scale(0.7)
        self.play(FadeIn(zero), FadeIn(large), Write(label_1024))
        self.play(large.animate.move_to(zero.get_center() + RIGHT*0.1))

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.wait(2)
