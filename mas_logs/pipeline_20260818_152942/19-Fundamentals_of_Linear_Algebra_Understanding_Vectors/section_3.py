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
        self.setup_layout("Scalar Multiplication", [
            "Scalars change a vector's length or direction.",
            "Multiplying by a scalar scales the vector.",
            "Negative scalars reverse the vector's direction."
        ])
        
        # Vector Setup
        origin = ORIGIN
        v_coords = np.array([1, 1, 0])
        vector = Arrow(origin, v_coords, color=WHITE)
        label_v = MathTex(r"\\vec{v}").next_to(vector.get_end(), UP)
        
        v_group = VGroup(vector, label_v)
        # Fix 26/37: Update position
        self.place_in_area(v_group, 'B2', 'D4', scale_factor=0.8)
        
        # Assets (Using placeholders as per instruction)
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(vector), Write(label_v))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        v2 = Arrow(origin, 2 * v_coords, color="#FFFF00")
        label_2v = MathTex(r"2\\vec{v}").next_to(v2.get_end(), UP)
        v2_group = VGroup(v2, label_2v)
        # Fix 27/37: Update position
        self.place_in_area(v2_group, 'B4', 'D6', scale_factor=0.8)
        
        self.play(ReplacementTransform(vector.copy(), v2), Write(label_2v))
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        v_neg = Arrow(origin, -1 * v_coords, color="#FF00FF")
        label_neg_v = MathTex(r"-\\vec{v}").next_to(v_neg.get_end(), DOWN)
        v_neg_group = VGroup(v_neg, label_neg_v)
        # Fix 28/37: Update position
        self.place_in_area(v_neg_group, 'D2', 'F4', scale_factor=0.8)
        
        self.play(ReplacementTransform(vector.copy(), v_neg), Write(label_neg_v))
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.wait(2)
