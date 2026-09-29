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
            "Convolution is a sliding window operation.",
            "An input signal meets a kernel filter.",
            "We perform element-wise multiplication and summation."
        ]
        self.setup_layout("Intuitive Hook: The 'Moving Window' Concept", lecture_lines)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pixel.svg]
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/window.svg]
        
        # Grid of pixels
        input_grid = VGroup()
        for i in range(5):
            for j in range(5):
                pixel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pixel.svg", color=WHITE)
                pixel.scale(0.15)
                pixel.move_to(self.grid["A4"] + np.array([j * 0.7, -i * 0.7, 0]))
                input_grid.add(pixel)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(input_grid))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/window.svg]
        kernel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/window.svg", color="#FF00FF")
        self.place_at_grid(kernel, "C5", scale_factor=0.6)
        
        self.play(Create(kernel))
        self.lecture[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        # Highlight pixels
        highlighted = VGroup(*[input_grid[i] for i in [6, 7, 8, 11, 12, 13, 16, 17, 18]])
        self.play(highlighted.animate.set_color("#00FFFF"))
        self.lecture[2].set_color("#00FFFF")
        
        # Animate shift
        self.play(kernel.animate.shift(RIGHT * 0.7))
        self.lecture[2].set_color("#FF00FF")
        
        # Scan animation
        self.play(kernel.animate.shift(LEFT * 0.7), run_time=0.5)
        self.lecture[2].set_color("#FFFF00")
