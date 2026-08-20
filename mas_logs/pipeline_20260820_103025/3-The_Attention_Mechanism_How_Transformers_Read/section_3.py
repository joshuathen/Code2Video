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
            "Query is your search term.",
            "Key is the label on a folder.",
            "Value is the content inside.",
            "Dot-product measures the similarity score.",
            "Softmax normalizes scores into probabilities."
        ]
        self.setup_layout("The Mathematical Architecture: Q, K, and V", lecture_lines)
        
        # Consistent color scheme
        color_q = "#FF4500"
        color_k = "#32CD32"
        color_v = "#1E90FF"
        
        q_label = Text("Q", color=color_q, font_size=36)
        k_label = Text("K", color=color_k, font_size=36)
        v_label = Text("V", color=color_v, font_size=36)
        
        searchbar = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/searchbar.svg")
        doc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/document.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_q)
        self.place_at_grid(q_label, 'B2', scale_factor=0.8)
        self.place_at_grid(searchbar, 'B5', scale_factor=0.5)
        self.play(FadeIn(q_label), FadeIn(searchbar))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color_k)
        self.place_at_grid(k_label, 'B3', scale_factor=0.8)
        self.play(FadeIn(k_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(color_v)
        self.place_at_grid(v_label, 'B4', scale_factor=0.8)
        self.play(FadeIn(v_label))
        
        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        dot_product_text = MathTex(r"Score = Q \cdot K^T", font_size=32)
        self.place_in_area(dot_product_text, 'C2', 'C4', scale_factor=0.7)
        self.play(Write(dot_product_text))
        
        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        softmax_text = MathTex(r"Softmax(Score)", font_size=32)
        final_output_label = Text("Attention Output", color="#00FFFF", font_size=28)
        
        self.place_in_area(softmax_text, 'D2', 'D4', scale_factor=0.7)
        self.place_at_grid(doc_icon, 'E5', scale_factor=0.5)
        self.place_at_grid(final_output_label, 'E3', scale_factor=0.7)
        
        self.play(Write(softmax_text), FadeIn(doc_icon), FadeIn(final_output_label))
        
        self.wait(2)
