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
        lines = [
            "We represent words as mathematical vectors.",
            "Dot products measure the similarity between words.",
            "Higher similarity means higher attention weight.",
            "Attention is a weighted sum of values.",
            "The formula scales values by dot products."
        ]
        self.setup_layout("Visualizing the Calculation", lines)

        # Prepare visual elements
        q_vec = Arrow(ORIGIN, UP*1.5, color=BLUE).set_length(1.5)
        k_vec = Arrow(ORIGIN, RIGHT*1.5, color=RED).set_length(1.5)
        vec_group = VGroup(q_vec, k_vec)
        
        # Asset placeholders (none.svg does not exist, using Dot as representative)
        asset_placeholder = Dot(radius=0.1, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(vec_group, 'C1', 'E2', scale_factor=0.6)
        self.play(Create(vec_group), FadeIn(asset_placeholder.move_to(self.grid['B1'])))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        dot_product_line = DashedLine(q_vec.get_end(), k_vec.get_end(), color="#FFFF00")
        self.play(Create(dot_product_line))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        weight_text = Text("Similarity", font_size=24, color="#00FF00")
        self.place_at_grid(weight_text, 'E3', scale_factor=0.7)
        self.play(Write(weight_text))
        self.lecture[2].set_color("#00FF00")

        # === Animation for Lecture Line 4 ===
        formula = MathTex(r"\\sum \\alpha_i v_i", color="#FF00FF")
        self.place_in_area(formula, 'B3', 'C5', scale_factor=0.8)
        self.play(FadeIn(formula))
        self.lecture[3].set_color("#FF00FF")

        # === Animation for Lecture Line 5 ===
        # Add placeholder for the final step
        final_score = Text("Score: 0.9", font_size=24, color=WHITE)
        self.place_at_grid(final_score, 'E5', scale_factor=0.8)
        self.play(FadeIn(final_score), FadeIn(asset_placeholder.copy().move_to(self.grid['E6'])))
        self.play(formula.animate.set_color(WHITE))
        self.lecture[4].set_color("#FFFFFF")
        
        self.wait(2)
