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
        lecture_lines = ["We calculate attention using Q, K, and V.", "Dot-product Q and K creates a heatmap.", "Softmax makes the focus sharper."]
        self.setup_layout("The Mechanism in Motion: Scaled Dot-Product Attention", lecture_lines)
        
        # Define colors for lecture lines
        c1, c2, c3 = "#FFD700", "#00FFFF", "#FF69B4"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(c1)
        q = Text("Q", color=c1).scale(1.2)
        k = Text("K", color=c2).scale(1.2)
        v = Text("V", color=c3).scale(1.2)
        qkv = VGroup(q, k, v).arrange(RIGHT, buff=0.5)
        # Applying fix for Issue 30: Use scale_factor 0.6 at B2
        self.place_at_grid(qkv, 'B2', scale_factor=0.6)
        
        # Add placeholder icon for Asset
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_at_grid(icon, 'A4', scale_factor=0.5)
        
        self.play(Write(qkv), FadeIn(icon))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(c2)
        # Applying fix for Issue 31: Use area B3-D5 with scale 0.9
        heatmap = Square(fill_opacity=0.5, color=c2).set_fill(color=c2)
        self.place_in_area(heatmap, 'B3', 'D5', scale_factor=0.9)
        
        self.play(FadeIn(heatmap))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(c3)
        # Applying fix for Issue 32: Use D6 with scale 0.6
        softmax_bar = BarChart([0.1, 0.7, 0.2], bar_colors=[c3], y_range=[0, 1, 0.1])
        self.place_at_grid(softmax_bar, 'D6', scale_factor=0.6)
        
        # Add placeholder icon for Asset
        icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_at_grid(icon2, 'E6', scale_factor=0.3)
        
        self.play(Create(softmax_bar), FadeIn(icon2))
        self.wait(2)
