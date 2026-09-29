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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary & Quick Check", [
            "Multiply two terms? Use the Product Rule.",
            "One function inside another? Use Chain Rule.",
            "Check your understanding with our final quiz."
        ])
        
        # Formulas to display
        prod_rule = MathTex(r"d/dx [f(x)g(x)] = f'(x)g(x) + f(x)g'(x)", color="#FF9900")
        chain_rule = MathTex(r"d/dx [f(g(x))] = f'(g(x)) \cdot g'(x)", color="#FF9900")
        formula_group = VGroup(prod_rule, chain_rule).arrange(DOWN, buff=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF9900")
        self.place_in_area(formula_group, 'A3', 'C6', scale_factor=0.8)
        self.play(Write(prod_rule))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF9900")
        self.play(Write(chain_rule))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        ready_msg = Text("Ready to Practice?", color="#00FF00", font_size=40)
        self.place_at_grid(ready_msg, 'F6', scale_factor=0.9)
        self.play(FadeIn(ready_msg))
        self.play(Indicate(ready_msg))
        self.wait(2)
