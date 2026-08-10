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
        self.setup_layout("The Transition Matrix (The Bridge)", [
            "Define the transition matrix P.",
            "Columns are new basis in old.",
            "Equation: vector B equals P, vector C.",
            "P maps coordinates between bases.",
            "The bridge is built."
        ])
        
        # Colors for lines
        colors = [YELLOW, BLUE, GREEN, RED, ORANGE]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        # Include bridge icon
        bridge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        self.place_at_grid(bridge_icon, 'F3', scale_factor=0.5)
        self.play(FadeIn(bridge_icon))
        
        matrix_p = MathTex("P = \\begin{pmatrix} [u]_B & [v]_B \\end{pmatrix}", font_size=32)
        self.place_at_grid(matrix_p, 'B3', scale_factor=0.8)
        self.play(Write(matrix_p))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(colors[1])
        u_vec = MathTex("[u]_B = \\begin{pmatrix} 1 \\\\ 0 \\end{pmatrix}", color=BLUE, font_size=28)
        v_vec = MathTex("[v]_B = \\begin{pmatrix} 0 \\\\ 1 \\end{pmatrix}", color=GREEN, font_size=28)
        self.place_at_grid(u_vec, 'B2', scale_factor=0.9)
        self.place_at_grid(v_vec, 'B4', scale_factor=0.9)
        self.play(FadeIn(u_vec), FadeIn(v_vec))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(colors[2])
        equation = MathTex("[v]_B = P [v]_C", font_size=36)
        self.place_in_area(equation, 'C2', 'C5', scale_factor=1.0)
        self.play(Write(equation))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(colors[3])
        map_arrow = Arrow(start=self.grid['D3'], end=self.grid['E3'], color=RED)
        self.play(GrowArrow(map_arrow))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(colors[4])
        bridge_text = Text("Bridge Complete", color=ORANGE, font_size=24)
        self.place_at_grid(bridge_text, 'D3', scale_factor=1.0)
        self.play(Write(bridge_text))
        self.wait(2)
