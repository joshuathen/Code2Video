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
        # Initialization
        title = "Introduction: The Geometric Hook"
        lines = [
            "Meet vectors v and w in three-dimensional space.",
            "Their cross product is a vector perpendicular to both.",
            "This unique operation reveals deep geometric properties."
        ]
        self.setup_layout(title, lines)

        # Vector Colors
        color_v = "#58C4DD"
        color_w = "#83C167"
        color_n = "#F8B195"

        # === Animation for Lecture Line 1 ===
        # Highlight current line
        self.play(self.lecture[0].animate.set_color(color_v))
        
        # Define origin at D3
        origin = self.grid["D3"]
        
        # Asset: common origin point
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/null.svg]
        origin_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/null.svg")
        self.place_at_grid(origin_icon, "D3", scale_factor=0.2)
        self.add(origin_icon)
        
        # Vector v and w
        v_arrow = Arrow(origin, self.grid["D5"], color=color_v, buff=0)
        w_arrow = Arrow(origin, self.grid["B4"], color=color_w, buff=0)
        
        # Fixed: Replaced MathTex with Text to avoid LaTeX dependency error
        v_label = Text("v", color=color_v, slant=ITALIC)
        w_label = Text("w", color=color_w, slant=ITALIC)
        
        # Fixes from issues 24 and 26
        self.place_at_grid(v_label, 'C5', scale_factor=0.8)
        self.place_at_grid(w_label, 'A4', scale_factor=0.8)
        
        self.play(Create(v_arrow), Write(v_label))
        self.play(Create(w_arrow), Write(w_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight current line and revert previous
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_n)
        )
        
        # Cross product vector n (perpendicular)
        n_arrow = Arrow(origin, self.grid["B2"], color=color_n, buff=0)
        
        self.play(Create(n_arrow))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight current line and revert previous
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_n)
        )
        
        # Fixed: Replaced MathTex with Text to avoid LaTeX dependency error
        cross_label = Text("v × w", color=color_n, slant=ITALIC)
        # Fix from issue 25
        self.place_in_area(cross_label, 'A1', 'B2', scale_factor=0.8)
        
        self.play(Write(cross_label))
        self.wait(2)
