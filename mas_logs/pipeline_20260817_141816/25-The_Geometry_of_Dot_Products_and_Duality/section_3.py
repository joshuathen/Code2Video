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
        self.setup_layout("The Concept of Duality", [
            "View dot product as a linear function.", 
            "Fixed vector 'v' defines output scalar.", 
            "Visualize parallel lines or level sets."
        ])
        
        # Setup visual elements
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        func_text = MathTex(r"L(x) = \vec{v} \cdot x", color=WHITE)
        self.place_in_area(func_text, 'B4', 'B6', scale_factor=0.9)
        
        vector_v = Vector(RIGHT + UP, color=YELLOW)
        label_v = MathTex(r"\vec{v}^*", color="#00FFFF").next_to(vector_v.get_end(), UP)
        v_group = VGroup(vector_v, label_v)
        self.place_at_grid(v_group, 'D4', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(Write(func_text))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        self.play(Create(v_group))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        lines = VGroup(*[Line(LEFT*1, RIGHT*1, color=BLUE_D).shift(UP*i*0.3) for i in range(-2, 3)])
        self.place_in_area(lines, 'D2', 'E5', scale_factor=1.0)
        self.play(FadeIn(lines))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
