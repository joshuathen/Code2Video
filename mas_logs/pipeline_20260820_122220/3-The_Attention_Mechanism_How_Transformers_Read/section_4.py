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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Attention combines Q, K, and V vectors.",
            "We calculate weights via dot products.",
            "Values are summed based on these weights.",
            "Result: A context-aware representation.",
            "The formula represents a weighted mix."
        ]
        self.setup_layout("The Mathematical Workflow", lecture_lines)

        # Define color scheme
        Q_COLOR = BLUE
        K_COLOR = GREEN
        V_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        # Attention combines Q, K, and V vectors.
        self.lecture[0].set_color(WHITE)
        
        # Using placeholder SVG
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        q_rect = Rectangle(width=0.8, height=1.5, color=Q_COLOR, fill_opacity=0.5)
        k_rect = Rectangle(width=0.8, height=1.5, color=K_COLOR, fill_opacity=0.5)
        v_rect = Rectangle(width=0.8, height=1.5, color=V_COLOR, fill_opacity=0.5)
        
        q_label = Text("Q", color=Q_COLOR).scale(0.7)
        k_label = Text("K", color=K_COLOR).scale(0.7)
        v_label = Text("V", color=V_COLOR).scale(0.7)
        q_label.next_to(q_rect, UP)
        k_label.next_to(k_rect, UP)
        v_label.next_to(v_rect, UP)
        
        group = VGroup(q_rect, k_rect, v_rect, q_label, k_label, v_label, icon)
        self.place_in_area(group, "A1", "C6", scale_factor=0.5)
        self.play(FadeIn(group))

        # === Animation for Lecture Line 2 ===
        # We calculate weights via dot products.
        self.lecture[1].set_color(ORANGE)
        dot_product_text = MathTex("Q \\cdot K^T", color=ORANGE)
        self.place_at_grid(dot_product_text, "D3")
        self.play(Write(dot_product_text))

        # === Animation for Lecture Line 3 ===
        # Values are summed based on these weights.
        self.lecture[2].set_color(YELLOW)
        sum_text = MathTex("\\sum (Weights \\cdot V)", color=YELLOW)
        self.place_at_grid(sum_text, "D4")
        self.play(Write(sum_text))

        # === Animation for Lecture Line 4 ===
        # Result: A context-aware representation.
        self.lecture[3].set_color(PURPLE)
        res_rect = Rectangle(width=1.5, height=0.8, color=PURPLE, fill_opacity=0.7)
        res_label = Text("Context-Aware\nOutput", font_size=16, color=WHITE).move_to(res_rect)
        res_group = VGroup(res_rect, res_label)
        self.place_at_grid(res_group, "D5", scale_factor=0.8)
        self.play(FadeIn(res_group))

        # === Animation for Lecture Line 5 ===
        # The formula represents a weighted mix.
        self.lecture[4].set_color(RED)
        formula = MathTex("Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V", font_size=24, color=RED)
        
        icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg").scale(0.3)
        formula_group = VGroup(formula, icon2).arrange(DOWN)
        
        self.place_in_area(formula_group, "E1", "F6", scale_factor=0.75)
        self.play(Write(formula_group))
        self.wait(2)
