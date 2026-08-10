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
            "Dot product measures similarity between two vectors.",
            "Query multiplied by Key gives raw attention scores.",
            "High scores mean more focus on that word.",
            "'It' pays attention to 'cat' over 'mat'.",
            "This creates our attention heatmap."
        ]
        self.setup_layout("The Mechanism: Calculating Attention Scores", lecture_lines)
        
        # Asset Loading
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")
        mat_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mat.svg")

        # === Animation for Lecture Line 1 ===
        formula = MathTex(r"\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V", color=WHITE)
        self.place_in_area(formula, 'A2', 'B5', scale_factor=0.75)
        self.place_at_grid(cat_icon, 'C3', scale_factor=0.3)
        self.play(Write(formula), FadeIn(cat_icon))
        self.play(self.lecture[0].animate.set_color(WHITE))

        # === Animation for Lecture Line 2 ===
        highlight = MathTex(r"\frac{QK^T}{\sqrt{d_k}}", color="#FF5733")
        self.place_at_grid(highlight, 'D3', scale_factor=1.0)
        self.play(FadeIn(highlight))
        self.play(self.lecture[1].animate.set_color("#FF5733"))

        # === Animation for Lecture Line 3 ===
        score_viz = Text("Scores", color="#33FF57").scale(0.8)
        self.place_at_grid(score_viz, 'D2', scale_factor=0.7)
        self.play(ReplacementTransform(highlight, score_viz))
        self.play(self.lecture[2].animate.set_color("#33FF57"))

        # === Animation for Lecture Line 4 ===
        weight_viz = Text("Weighted Values", color="#3357FF").scale(0.8)
        self.place_at_grid(weight_viz, 'E2', scale_factor=0.7)
        self.play(FadeIn(weight_viz))
        self.play(self.lecture[3].animate.set_color("#3357FF"))

        # === Animation for Lecture Line 5 ===
        final_block = Rectangle(color="#CCCCCC", width=4, height=1.5)
        label = Text("Attention Heatmap", color="#CCCCCC").scale(0.6)
        group = VGroup(final_block, label, mat_icon)
        self.place_at_grid(mat_icon, 'F6', scale_factor=0.3)
        self.place_in_area(group, 'F2', 'F5', scale_factor=0.7)
        self.play(Create(final_block), Write(label), FadeIn(mat_icon))
        self.play(self.lecture[4].animate.set_color("#CCCCCC"))
        
        self.wait(2)
