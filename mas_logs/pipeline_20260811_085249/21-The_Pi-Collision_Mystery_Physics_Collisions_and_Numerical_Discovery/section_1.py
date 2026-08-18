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
            "Consider two blocks on a frictionless surface.",
            "One small block, one massive block, and a wall.",
            "They exchange momentum via perfectly elastic collisions.",
            "Kinetic energy is conserved in every impact.",
            "This simple system holds a hidden mathematical secret."
        ]
        self.setup_layout("The Setup: The Elastic Collision Paradox", lecture_lines)
        
        # Load Assets
        block_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg"
        wall_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg"
        
        wall = SVGMobject(wall_svg)
        block_small = SVGMobject(block_svg)
        block_large = SVGMobject(block_svg)
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(block_small, 'C2', scale_factor=0.3)
        self.place_at_grid(block_large, 'C3', scale_factor=0.6)
        self.play(FadeIn(block_small), FadeIn(block_large))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(wall, 'C6', scale_factor=1.0)
        self.play(FadeIn(wall))
        self.lecture[1].set_color(RED)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(block_small.animate.shift(RIGHT * 1.5))
        self.lecture[2].set_color(GREEN)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(block_small.animate.shift(LEFT * 1.0))
        self.lecture[3].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(Indicate(block_small), Indicate(block_large))
        self.lecture[4].set_color(PURPLE)
        self.wait(2)
