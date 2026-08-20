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
            "Energy and momentum are both conserved.",
            "Velocity vectors dictate collision counts.",
            "The wall reverses the direction of velocity."
        ]
        self.setup_layout("Prerequisite Knowledge: Conservation Laws", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Momentum: p = mv, Energy: E = 0.5mv^2
        # Use #00FF00 for variables
        
        # Load asset
        wall_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")
        self.place_at_grid(wall_icon, "B3", scale_factor=0.5)
        
        p_eq = MathTex("p = m v", font_size=36)
        p_eq.set_color_by_tex("p", "#00FF00")
        p_eq.set_color_by_tex("m", "#00FF00")
        p_eq.set_color_by_tex("v", "#00FF00")
        
        e_eq = MathTex(r"E = 0.5 m v^2", font_size=36)
        e_eq.set_color_by_tex("E", "#00FF00")
        e_eq.set_color_by_tex("m", "#00FF00")
        e_eq.set_color_by_tex("v", "#00FF00")
        
        # Group equations as per critique
        formula_group = VGroup(p_eq, e_eq).arrange(RIGHT, buff=1.0)
        self.place_in_area(formula_group, "C1", "D6", scale_factor=0.75)
        
        self.play(FadeIn(wall_icon), FadeIn(formula_group))
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 2 ===
        # Conservation equations
        cons_p = MathTex("m_1 v_1 + m_2 v_2 = \\text{const}", font_size=30)
        cons_e = MathTex("0.5 m_1 v_1^2 + 0.5 m_2 v_2^2 = \\text{const}", font_size=30)
        cons_group = VGroup(cons_p, cons_e).arrange(DOWN, buff=0.5)
        
        self.place_in_area(cons_group, 'E1', 'F6', scale_factor=0.7)
        
        self.play(FadeIn(cons_group))
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # === Animation for Lecture Line 3 ===
        # Velocity reversing at wall
        dot = Dot(color=YELLOW)
        dot.move_to(self.grid["A3"])
        vec = Arrow(start=ORIGIN, end=LEFT, color=RED).next_to(dot, LEFT, buff=0.1)
        
        self.add(dot, vec)
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.wait(2)
