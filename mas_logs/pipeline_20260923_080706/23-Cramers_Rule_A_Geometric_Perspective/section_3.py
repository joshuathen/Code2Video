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
            "Replacing a column creates new parallelograms.",
            "Variable equals new area over old area.",
            "This ratio solves for each coordinate.",
            "Cramer's Rule is geometric scaling.",
            "It visualizes linear system solutions."
        ]
        self.setup_layout("Geometric Derivation of Cramer's Rule", lecture_lines)

        # Base vectors for visual
        orig_v1 = np.array([1, 2, 0])
        orig_v2 = np.array([2, 0.5, 0])
        
        # Colors
        c1 = BLUE
        c2 = GREEN
        b_col = YELLOW
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(c1))
        # Visuals: Showing the original parallelogram and then the replaced one
        vec_v1 = Arrow(ORIGIN, orig_v1, color=c1, buff=0)
        vec_v2 = Arrow(ORIGIN, orig_v2, color=c2, buff=0)
        para = Polygon(ORIGIN, orig_v1, orig_v1+orig_v2, orig_v2, fill_opacity=0.3, color=WHITE)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg]
        para_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg")
        para_group = VGroup(vec_v1, vec_v2, para, para_icon)
        
        self.place_in_area(para_group, 'A4', 'C6', scale_factor=0.6)
        para_group.set_color("#FF00FF") # As per requirement for the transformed/replaced icon area
        
        self.play(Create(vec_v1), Create(vec_v2), FadeIn(para), FadeIn(para_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(c2))
        # Visuals: Show ratio area
        label_ratio = MathTex(r"x = \frac{\text{Area}_{new}}{\text{Area}_{orig}}", font_size=24)
        
        # Fix 28/29: Move formula
        self.place_at_grid(label_ratio, 'D5', scale_factor=0.8)
        self.play(Write(label_ratio))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(b_col))
        self.play(Indicate(label_ratio))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(RED))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        self.wait(2)
