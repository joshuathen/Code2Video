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
        lecture_lines = ["Self-attention weights word importance.", "Words connect with varying strength.", "'It' connects strongly to 'animal'."]
        self.setup_layout("The Transformer Core: Self-Attention Mechanism", lecture_lines)
        
        # --- Preparation ---
        formula = MathTex(r"Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}}\right)V", font_size=32, color=WHITE)
        animal_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/animal.svg")
        
        # Place formula and icon in defined area
        group = VGroup(formula, animal_icon).arrange(DOWN, buff=0.3)
        self.place_in_area(group, 'A3', 'B5', scale_factor=0.8)

        # Matrices representation
        q_label = Text("Q", font_size=24, color=WHITE)
        k_label = Text("K", font_size=24, color=WHITE)
        v_label = Text("V", font_size=24, color=WHITE)
        matrices = VGroup(q_label, k_label, v_label).arrange(RIGHT, buff=0.5)
        self.place_in_area(matrices, 'C2', 'C5', scale_factor=0.8)
        
        # Heatmap area
        heatmap = Square(side_length=2.0, color=BLUE, fill_opacity=0.3)
        self.place_in_area(heatmap, 'D3', 'F5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(Write(formula), FadeIn(animal_icon))
        self.lecture[0].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(matrices))
        self.play(matrices.animate.set_color("#FFD700"), run_time=0.5)
        self.play(matrices.animate.set_color(WHITE), run_time=0.5)
        self.lecture[1].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(heatmap.animate.set_color("#FF0000"))
        self.lecture[2].set_color("#FF0000")
        self.wait(2)
