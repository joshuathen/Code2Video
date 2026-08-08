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
        self.setup_layout("Prerequisite Physics: Conservation Laws", [
            "Collisions must conserve total momentum and energy.",
            "Massive blocks act like moving walls.",
            "Small blocks are forced to reverse direction."
        ])
        
        # === Animation for Lecture Line 1 ===
        formula = MathTex(r"p_1 + p_2 = \text{const}", font_size=36, color=WHITE)
        self.place_at_grid(formula, 'B2')
        self.play(Write(formula))
        self.play(formula.animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg]
        big_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color="#FFFF00")
        self.place_at_grid(big_block, 'C4', scale_factor=1.5)
        self.play(FadeIn(big_block))
        
        # Move like a wall
        self.play(big_block.animate.shift(LEFT * 1.5), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg]
        small_block = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg", color="#00FFFF")
        self.place_at_grid(small_block, 'D5', scale_factor=0.5)
        self.add(small_block)
        
        # Reverse direction
        self.play(small_block.animate.shift(RIGHT * 1.0), run_time=1)
        self.wait(1)
