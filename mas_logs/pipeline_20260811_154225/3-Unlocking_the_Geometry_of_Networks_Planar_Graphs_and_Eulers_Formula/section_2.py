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
        self.setup_layout("Euler's Characteristic Formula", [
            "Euler's formula relates V, E, and F.",
            "The formula is V minus E plus F.",
            "For planar graphs, this equals two.",
            "Simple shapes like triangles prove this.",
            "Every connected graph obeys this rule."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg]
        formula = MathTex("V - E + F = 2", font_size=48)
        self.place_in_area(formula, 'B4', 'B5', scale_factor=0.8)
        self.play(Write(formula))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        v_part = formula[0][0]
        e_part = formula[0][2]
        f_part = formula[0][4]
        self.play(
            v_part.animate.set_color(RED),
            e_part.animate.set_color(GREEN),
            f_part.animate.set_color(BLUE)
        )
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        self.play(formula.animate.scale(0.7).move_to(self.grid['A6']))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        # Using SVG asset
        triangle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/triangle.svg")
        triangle.set_color("#33FF57")
        self.place_in_area(triangle, 'D3', 'E4', scale_factor=1.0)
        
        v_label = Text("V=3", font_size=20, color=WHITE).scale(0.7)
        e_label = Text("E=3", font_size=20, color=WHITE).scale(0.7)
        f_label = Text("F=2", font_size=20, color=WHITE).scale(0.7)
        labels = VGroup(v_label, e_label, f_label).arrange(DOWN)
        labels.next_to(triangle, RIGHT)
        
        self.play(Create(triangle), Write(labels))
        self.lecture[3].set_color(YELLOW)

        # === Animation for Lecture Line 5 ===
        calc = MathTex("3 - 3 + 2 = 2", font_size=36, color="#FF5733")
        calc.next_to(triangle, DOWN)
        self.play(Write(calc))
        self.play(Indicate(formula), Indicate(calc))
        self.lecture[4].set_color(YELLOW)
        self.wait(2)
