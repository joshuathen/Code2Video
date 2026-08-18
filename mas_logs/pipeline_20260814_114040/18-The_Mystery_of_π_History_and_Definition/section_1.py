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
        self.setup_layout("The Mystery of \u03c0: The Constant Ratio", [
            "Every circle has a special, constant ratio.",
            "It links the circle's circumference and diameter.",
            "We call this special ratio pi.",
            "Size doesn't change this constant relationship.",
            "Pi is truly universal for all circles."
        ])
        
        # Assets
        # Loading SVG manually for local simulation
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        c_label = Text("C", font_size=24, color=WHITE)
        d_line = Line(start=LEFT*0.5, end=RIGHT*0.5, color="#FF00FF")
        d_label = Text("d", font_size=24, color="#FF00FF")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.place_at_grid(circle, 'C4', scale_factor=1.2)
        c_label.next_to(circle, UP, buff=0.1)
        self.play(Create(circle), Write(c_label))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        d_line.move_to(circle.get_center())
        d_label.next_to(d_line, DOWN, buff=0.1)
        self.play(Create(d_line), Write(d_label))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        pi_text = MathTex(r"\pi = \frac{C}{d}", color="#FFFFFF")
        self.place_in_area(pi_text, 'C4', 'D5', scale_factor=1.2)
        self.play(Write(pi_text))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FFFF")
        self.play(Indicate(pi_text))
        self.wait(2)
