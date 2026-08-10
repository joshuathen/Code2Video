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
            "Convolution combines input and kernel matrices.",
            "Multiply overlapping elements, then sum them up.",
            "Flipping the kernel defines the operation formally.",
            "Slide across grid to create new map.",
            "One input pixel results from every shift."
        ]
        self.setup_layout("Mathematical Core: Element-wise Product and Sum", lecture_lines)

        # Background Asset
        bg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(bg, "A1", "F6", scale_factor=0.5)
        self.add_to_back(bg) if hasattr(self, 'add_to_back') else self.add(bg)

        # Assets
        vec_a = MathTex("A = [1, 2, 3]", color=BLUE)
        vec_k = MathTex("K = [0.5, 1, 2]", color=RED)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.place_at_grid(vec_a, "B3", scale_factor=0.9)
        self.place_at_grid(vec_k, "C3", scale_factor=0.9)
        self.play(Write(vec_a), Write(vec_k))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        # Showing product elements
        arrows = VGroup(*[Arrow(vec_a[0][i+2].get_bottom(), vec_k[0][i+2].get_top(), color=WHITE, buff=0.1) for i in range(3)])
        self.play(Create(arrows))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        # Just highlighting kernel
        self.play(Indicate(vec_k))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN)
        # Visual slide simulation
        rect = Square(side_length=0.5, color=GREEN).next_to(vec_a, RIGHT)
        self.play(FadeIn(rect))
        self.play(rect.animate.shift(RIGHT))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PURPLE)
        res = MathTex("= 0.5 + 2 + 6 = 8.5", color=PURPLE)
        self.place_at_grid(res, "D5", scale_factor=0.9)
        self.play(Write(res))
        self.wait(1)
