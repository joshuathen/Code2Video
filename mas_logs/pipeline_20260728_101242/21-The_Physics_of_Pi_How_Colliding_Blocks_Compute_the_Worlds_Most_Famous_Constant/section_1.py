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
        # Data from storyboard and outline
        title_text = "The Setup: A Frictionless World"
        lecture_lines = [
            "Imagine a frictionless floor with a wall.",
            "Small block m sits still near the wall.",
            "Massive block M slides in from the right."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # "Imagine a frictionless floor with a wall."
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Define floor line and wall relative to grid
        floor_y = -3.3
        floor = Line(
            start=[0.0, floor_y, 0], 
            end=[6.0, floor_y, 0], 
            color=WHITE
        )
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg]
        wall = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wall.svg")
        # Place wall on the left side. Grid col 1 center is 0.5.
        self.place_in_area(wall, "A1", "F1", scale_factor=1.0)
        # Shift it to the left edge of the grid area
        wall.shift(LEFT * 0.5)
        
        self.play(Create(floor), FadeIn(wall))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Small block m sits still near the wall."
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Block m: small square (side 0.6)
        block_m = Square(side_length=0.6, fill_opacity=0.8, color="#00FF00")
        self.place_at_grid(block_m, "F1")
        # Shift to align bottom with floor_y = -3.3
        block_m.shift(DOWN * 0.2)
        
        # Reposition label_m to D1 (Issue 22)
        label_m = MathTex("m", color="#00FF00")
        self.place_at_grid(label_m, "D1")
        
        self.play(FadeIn(block_m), Write(label_m))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Massive block M slides in from the right."
        self.play(self.lecture[2].animate.set_color("#0000FF"))
        
        # Block M: massive square (side 1.2)
        block_M = Square(side_length=1.2, fill_opacity=0.8, color="#0000FF")
        self.place_at_grid(block_M, "F6")
        # Shift to align bottom with floor_y = -3.3
        block_M.shift(UP * 0.1)
        
        # Reposition label_M to D6 (Issue 21)
        label_M = MathTex("M", color="#0000FF")
        self.place_at_grid(label_M, "D6")
        
        # Yellow arrow for velocity
        v_arrow = Arrow(
            start=RIGHT * 0.8, 
            end=LEFT * 0.8, 
            color="#FFFF00", 
            buff=0,
            max_tip_length_to_length_ratio=0.3
        )
        v_arrow.next_to(block_M, LEFT, buff=0.1)
        
        self.play(FadeIn(block_M), Write(label_M))
        self.play(Create(v_arrow))
        
        # Animate block M group sliding left
        m_group = VGroup(block_M, label_M, v_arrow)
        target_x = 3.5
        current_x = m_group.get_center()[0]
        shift_vec = [target_x - current_x, 0, 0]
        
        self.play(
            m_group.animate.shift(shift_vec), 
            run_time=2, 
            rate_func=linear
        )
        
        # Final visual: display 'Elastic Collisions' and pulse blocks
        # Reposition elastic_label to A2-B5 with scale 0.6 (Issue 20)
        elastic_label = Text("Elastic Collisions", color="#FFA500", font_size=32)
        self.place_in_area(elastic_label, "A2", "B5", scale_factor=0.6)
        
        self.play(
            Indicate(block_m, color="#FFA500", scale_factor=1.2),
            Indicate(block_M, color="#FFA500", scale_factor=1.2),
            Write(elastic_label)
        )
        self.wait(2)
