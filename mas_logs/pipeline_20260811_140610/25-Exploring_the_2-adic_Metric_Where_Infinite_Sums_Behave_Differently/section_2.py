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
            "The 2-adic valuation counts powers of 2.",
            "We define 2-adic absolute value as 2 power -v2(x).",
            "In this world, powers of 2 are tiny."
        ]
        self.setup_layout("The 2-adic Perspective: Defining the Valuation", lecture_lines)
        
        # Assets
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        magnifying = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying.svg")
        
        # === Animation for Lecture Line 1 ===
        # Display the integer n = 2^k * m
        formula = MathTex(r"n = 2^k \cdot m", color="#FFFFFF")
        self.place_in_area(formula, 'B4', 'B6', scale_factor=0.8) # B003: restricted to col 4-6
        self.place_at_grid(ruler, 'A4', scale_factor=0.5)
        self.play(Write(formula), FadeIn(ruler))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        # Isolate the power k as the 2-adic valuation v2(n)
        valuation = MathTex(r"v_2(n) = k", color="#00FFFF")
        self.place_at_grid(valuation, 'D4', scale_factor=0.8)
        self.play(FadeIn(valuation))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Animate k increasing/absolute value context
        abs_val = MathTex(r"|n|_2 = 2^{-v_2(n)}", color="#FF69B4")
        self.place_at_grid(abs_val, 'E4', scale_factor=0.8)
        self.place_at_grid(magnifying, 'F4', scale_factor=0.5)
        self.play(FadeIn(abs_val), FadeIn(magnifying))
        self.lecture[2].set_color("#FF69B4")
        
        self.wait(2)
