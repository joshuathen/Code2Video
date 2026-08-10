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
        lecture_lines = [
            "We treat retrieval like a search system.",
            "Queries ask for what we need.",
            "Keys label what is available.",
            "Values contain the actual information content."
        ]
        self.setup_layout("The Intuition: Query, Key, and Value (QKV)", lecture_lines)
        
        # Load SVGs
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg", color=WHITE)
        doc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/document.svg", color=WHITE)

        # Prepare vectors
        query = Arrow(ORIGIN, UP*1.0, color="#FF0000")
        key = Arrow(ORIGIN, RIGHT*1.0, color="#00FF00")
        value = Arrow(ORIGIN, LEFT*1.0, color="#0000FF")
        
        q_label = MathTex("Q", color="#FF0000")
        k_label = MathTex("K", color="#00FF00")
        v_label = MathTex("V", color="#0000FF")
        
        # Groupings
        q_group = VGroup(query, q_label, magnifying_glass)
        k_group = VGroup(key, k_label)
        v_group = VGroup(value, v_label, doc_icon)
        
        qkv_system = VGroup(q_group, k_group, v_group)
        self.place_in_area(qkv_system, 'B2', 'E5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF0000"))
        self.place_at_grid(q_label, 'B2', scale_factor=0.7, offset=DOWN*0.5)
        self.play(Create(query), Write(q_label), FadeIn(magnifying_glass))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        self.place_at_grid(k_label, 'B4', scale_factor=0.7, offset=DOWN*0.5)
        self.play(Create(key), Write(k_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#0000FF"))
        self.place_at_grid(v_label, 'B6', scale_factor=0.7, offset=DOWN*0.5)
        self.play(Create(value), Write(v_label), FadeIn(doc_icon))
        
        projection_line = DashedLine(query.get_end(), key.get_end(), color=WHITE)
        self.play(Create(projection_line))
        self.wait(2)
