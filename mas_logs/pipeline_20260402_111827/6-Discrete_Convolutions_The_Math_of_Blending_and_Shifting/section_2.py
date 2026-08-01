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
        # Setup specific to Section 2: Discrete Convolutions
        self.setup_layout(
            "Discrete Convolutions: The Math of Blending and Shifting",
            [
                "- Definition: (f * g)[n]",
                "- Sliding windows mechanism",
                "- Element-wise multiplication",
                "- Summation of products",
                "- Signal shifting"
            ]
        )

        # 1. Formula visualization - Replaced MathTex with Text to avoid LaTeX dependency
        formula = Text(
            "(f * g)[n] = Σ f[m]g[n-m]", 
            font_size=25, 
            color=YELLOW
        )
        self.place_at_grid(formula, "B3", scale_factor=0.9)
        
        # 2. Discrete sequence representation - Replaced MathTex with Text
        seq_f = Text("f = [1, 2, 3, 2, 1]", font_size=21)
        seq_g = Text("g = [0.5, 1, 0.5]", font_size=21)
        
        self.place_at_grid(seq_f, "D3")
        self.place_at_grid(seq_g, "E3")

        # Animations
        self.play(Write(self.title))
        self.play(FadeIn(self.lecture, shift=RIGHT))
        self.wait(0.5)
        
        self.play(Write(formula))
        self.wait(0.5)
        
        self.play(
            Create(seq_f),
            Create(seq_g)
        )
        
        # Highlight shifting concept
        box = SurroundingRectangle(seq_g, color=BLUE)
        self.play(Create(box))
        self.play(box.animate.shift(RIGHT * 0.5), run_time=1)
        self.play(box.animate.shift(LEFT * 0.5), run_time=1)
        
        self.wait(2)
