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
        lecture_lines = [
            "We need a robust formula.",
            "Euler helps us count accurately.",
            "Moser provides the correct formula.",
            "Combinations explain the regions perfectly.",
            "Six points equal thirty-one regions."
        ]
        self.setup_layout("The Combinatorial Formula", lecture_lines)
        
        # 0: We need a robust formula.
        formula = MathTex(r"p = \frac{n^2 + n + 2}{2}", font_size=40)
        # Fix VideoCritic #30: Move formula to C4-E6 area
        self.place_in_area(formula, 'C4', 'E6', scale_factor=0.9)
        self.play(Write(formula))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        
        # Add Asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/euler.svg]
        euler_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/euler.svg")
        # Fix VideoCritic #31: Move euler to B5
        self.place_at_grid(euler_icon, 'B5', scale_factor=0.7)
        self.play(FadeIn(euler_icon))
        
        self.wait(1)

        # 1: Euler helps us count accurately.
        euler = MathTex(r"V - E + F = 2", font_size=36, color=YELLOW)
        # Keep euler near the icon or other free space
        self.place_at_grid(euler, "B3")
        self.play(FadeIn(euler))
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)

        # 2: Moser provides the correct formula.
        self.play(Indicate(formula))
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.wait(1)

        # 3: Combinations explain the regions perfectly.
        # Highlight n^2 term in formula as requested by storyboard
        term_n2 = formula[0][2:4] # p = (n^2 + ...)
        highlight = SurroundingRectangle(term_n2, color="#00FFFF")
        
        self.play(Create(highlight))
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        self.wait(1)

        # 4: Six points equal thirty-one regions.
        calc = MathTex(r"n=6 \implies p = 31", font_size=32)
        # Fix VideoCritic #32: Move calculation to E5
        self.place_at_grid(calc, 'E5', scale_factor=0.8)
        self.play(Write(calc))
        self.play(self.lecture[4].animate.set_color("#00FF00"))
        self.wait(2)
