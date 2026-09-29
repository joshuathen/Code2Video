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
        lecture_lines = ["Generating functions turn counting into algebra.", 
                         "Roots of unity provide precise selection.", 
                         "Symmetry isolates the information we need."]
        self.setup_layout("Summary and Intuition", lecture_lines)
        
        # Elements
        filter_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/filter.svg", color=PURPLE)
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg", color=GOLD)
        
        arrow_in = Arrow(start=LEFT, end=RIGHT, color=WHITE).scale(0.5)
        arrow_out = Arrow(start=LEFT, end=RIGHT, color=WHITE).scale(0.5)
        
        input_label = MathTex(r"A(x)", color=WHITE)
        output_label = MathTex(r"a_k", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#87CEFA")
        self.place_at_grid(filter_icon, 'C3', scale_factor=0.6)
        self.place_at_grid(input_label, 'C2', scale_factor=0.8)
        self.place_at_grid(output_label, 'C4', scale_factor=0.8)
        arrow_in.next_to(filter_icon, LEFT)
        arrow_out.next_to(filter_icon, RIGHT)
        self.add(filter_icon, arrow_in, arrow_out, input_label, output_label)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#800080")
        root_formula = MathTex(r"\omega^k = e^{i 2\pi k / n}", color="#800080")
        self.place_at_grid(root_formula, 'B3', scale_factor=0.7)
        self.play(Write(root_formula))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        final_eq = MathTex(r"a_k = \frac{1}{n} \sum_{j=0}^{n-1} A(\omega^j) \omega^{-jk}", color="#FFD700")
        self.place_in_area(final_eq, 'D3', 'D5', scale_factor=0.7)
        self.place_at_grid(magnifying_glass, 'E4', scale_factor=0.5)
        self.play(FadeIn(final_eq), FadeIn(magnifying_glass))
        self.wait(3)
