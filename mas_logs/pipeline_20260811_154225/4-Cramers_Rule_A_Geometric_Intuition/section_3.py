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
            "Area of (b, v2) scaled yields x1.",
            "Replacing v1 with b shifts the geometry.",
            "The area ratio mirrors the scaling factor."
        ]
        self.setup_layout("Visualizing Cramer's Rule: The Area Ratio", lecture_lines)
        
        # Define base vectors for visualization
        v1 = np.array([1, 1, 0])
        v2 = np.array([0.5, 1, 0])
        b = np.array([1, 0.5, 0])
        
        # Assets placeholders
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        icon1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        icon3 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        # Parallelograms (Area)
        orig_para = Polygon([0,0,0], v1, v1+v2, v2, fill_opacity=0.3, color="#FF0000")
        new_para = Polygon([0,0,0], b, b+v2, v2, fill_opacity=0.3, color="#00FF00")
        
        # Labels
        label_a = MathTex(r"\det(A)", color="#FF0000")
        label_ax = MathTex(r"\det(A_x)", color="#00FF00")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF0000")
        self.place_in_area(orig_para, "A4", "B6", scale_factor=0.6)
        self.place_at_grid(label_a, "A3", scale_factor=0.7)
        self.play(Create(orig_para), Write(label_a), FadeIn(icon1.scale(0.5).next_to(label_a)))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        self.place_in_area(new_para, "D4", "E6", scale_factor=0.6)
        self.place_at_grid(label_ax, "D3", scale_factor=0.7)
        self.play(Create(new_para), Write(label_ax), FadeIn(icon2.scale(0.5).next_to(label_ax)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        ratio = MathTex(r"x_1 = \frac{\det(A_x)}{\det(A)}", color="#FFFF00")
        self.place_at_grid(ratio, "C5", scale_factor=0.9)
        self.play(Write(ratio), FadeIn(icon3.scale(0.5).next_to(ratio)))
        self.wait(2)
