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
            "Classical objects exist in definite, singular states.",
            "Quantum systems allow for multiple states simultaneously.",
            "State vectors represent these quantum states mathematically."
        ]
        self.setup_layout("The Classical vs. Quantum World", lecture_lines)
        
        # Assets
        ball_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg"
        
        # === Animation for Lecture Line 1 ===
        ball = SVGMobject(ball_path, color=WHITE)
        self.place_at_grid(ball, 'B2', scale_factor=0.7)
        self.add(ball)
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        wave = FunctionGraph(lambda x: 0.5 * np.sin(4 * x), x_range=[-1.5, 1.5], color="#00FF00")
        self.place_at_grid(wave, 'D3', scale_factor=0.6)
        self.add(wave)
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        vector = Arrow(start=ORIGIN, end=UP*1.0 + RIGHT*0.3, color="#FFFF00")
        label = MathTex(r"|\psi\rangle", color="#FFFFFF")
        
        # Position vector
        self.place_at_grid(vector, 'B5', scale_factor=0.8)
        
        # Position label relative to vector
        label.next_to(vector.get_end(), UP, buff=0.1)
        
        # Add extra ball for the description constraint
        ball2 = SVGMobject(ball_path, color=WHITE)
        self.place_at_grid(ball2, 'D5', scale_factor=0.4)
        
        self.add(vector, label, ball2)
        self.lecture[2].set_color("#FFFF00")
        
        self.wait(2)
