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
        self.setup_layout("The Core Engine: Self-Attention Mechanism", [
            "Self-attention acts as a relevance map.", 
            "Each word attends to all others simultaneously.", 
            "Vectors determine contextual dependencies between words.", 
            "Query, Key, and Value models interactions.", 
            "Attention highlights specific, relevant information pathways."
        ])
        
        # Define elements
        q_box = Square(side_length=1.0, color=WHITE)
        q_text = Text("Q", color=WHITE).move_to(q_box.get_center())
        q_group = VGroup(q_box, q_text)
        
        k_box = Square(side_length=1.0, color=WHITE)
        k_text = Text("K", color=WHITE).move_to(k_box.get_center())
        k_group = VGroup(k_box, k_text)
        
        v_box = Square(side_length=1.0, color=WHITE)
        v_text = Text("V", color=WHITE).move_to(v_box.get_center())
        v_group = VGroup(v_box, v_text)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_at_grid(q_group, 'B2', scale_factor=0.6)
        self.place_at_grid(k_group, 'B3', scale_factor=0.6)
        self.place_at_grid(v_group, 'B4', scale_factor=0.6)
        self.play(FadeIn(q_group), FadeIn(k_group), FadeIn(v_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF69B4")
        arrow1 = Arrow(q_group.get_center(), k_group.get_center(), color="#FF69B4")
        self.play(Create(arrow1))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFA500")
        heatmap = Rectangle(width=1.5, height=1.5, color="#FFA500", fill_opacity=0.3)
        self.place_at_grid(heatmap, 'D2', scale_factor=0.7)
        self.play(FadeIn(heatmap))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00CED1")
        weight_text = MathTex(r"\\sum w \\cdot V", color="#00CED1")
        self.place_at_grid(weight_text, 'D4', scale_factor=0.7)
        self.play(Write(weight_text))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        final_box = RoundedRectangle(corner_radius=0.2, color=WHITE)
        self.place_in_area(final_box, 'E2', 'E5', scale_factor=0.5)
        self.play(Create(final_box))
        self.wait(2)
