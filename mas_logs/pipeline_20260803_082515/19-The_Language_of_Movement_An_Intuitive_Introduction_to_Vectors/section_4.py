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
        # Data from storyboard
        title = "Vector Scaling: The Growth Potion"
        lines = [
            "Scaling changes a vector's size.",
            "Multiplying by two doubles its length.",
            "Negative multipliers reverse the vector's direction."
        ]
        self.setup_layout(title, lines)

        # Vector origin at D3
        origin = self.grid["D3"]
        scale_val = 0.35

        # === Animation for Lecture Line 1 ===
        # Display the original light blue vector v = [3, 4] (#00FFFF).
        vec_v_end = origin + np.array([3 * scale_val, 4 * scale_val, 0])
        
        vector_v = Arrow(origin, vec_v_end, buff=0, color="#00FFFF", stroke_width=6)
        label_v = MathTex("v", color="#00FFFF")
        # Issue 36 fix: Position label_v at C5
        self.place_at_grid(label_v, "C5", scale_factor=0.8)
        label_v.shift(LEFT * 0.3 + UP * 0.2) # Adjust shift for new grid position

        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.play(Create(vector_v), Write(label_v))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Multiplying by two doubles its length.
        vec_2v_end = origin + np.array([6 * scale_val, 8 * scale_val, 0])
        vector_2v = Arrow(origin, vec_2v_end, buff=0, color="#ADFF2F", stroke_width=6)
        label_2v = MathTex("2v", color="#ADFF2F")
        # Issue 37 fix: Position label_2v at B6
        self.place_at_grid(label_2v, "B6", scale_factor=0.8)
        label_2v.shift(LEFT * 0.2 + UP * 0.2) # Adjust shift for new grid position

        self.play(self.lecture[1].animate.set_color("#ADFF2F"))
        self.play(
            Transform(vector_v, vector_2v),
            Transform(label_v, label_2v)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Negative multipliers reverse the vector's direction.
        vec_neg_v_end = origin + np.array([-3 * scale_val, -4 * scale_val, 0])
        vector_neg_v = Arrow(origin, vec_neg_v_end, buff=0, color="#FF6347", stroke_width=6)
        label_neg_v = MathTex("-v", color="#FF6347")
        # Issue 38 fix: Position label_neg_v at E3
        self.place_at_grid(label_neg_v, "E3", scale_factor=0.8)
        label_neg_v.shift(LEFT * 0.4 + DOWN * 0.4) # Adjust shift for new grid position

        self.play(self.lecture[2].animate.set_color("#FF6347"))
        self.play(
            Transform(vector_v, vector_neg_v),
            Transform(label_v, label_neg_v)
        )
        self.wait(2)
