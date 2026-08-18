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
        lecture_lines = ["Replace a column of A with b.", "This creates a new, scaled parallelogram.", "The area ratio reveals the specific coordinate.", "This geometric change isolates each coordinate value.", "We define Cramer's Rule using these areas."]
        self.setup_layout("Geometric Derivation of Cramer's Rule", lecture_lines)
        
        para_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/parallelogram.svg"

        # Helpers
        def get_parallelogram():
            try:
                return SVGMobject(para_asset)
            except:
                return Square(side_length=1, color=WHITE, fill_opacity=0.5)

        # === Animation for Lecture Line 1 ===
        # Replace column a1 in matrix A with vector b to form A1.
        self.lecture[0].set_color(YELLOW)
        mat_text = MathTex(r"A = [\vec{a_1} | \vec{a_2}] \rightarrow A_1 = [\vec{b} | \vec{a_2}]").scale(0.8)
        self.place_in_area(mat_text, 'A4', 'B6')
        self.play(Write(mat_text))

        # === Animation for Lecture Line 2 ===
        # Show a new parallelogram formed by b and a2; color it #0000FF.
        self.lecture[1].set_color(BLUE)
        para1 = get_parallelogram()
        para1.set_fill(BLUE, opacity=0.6).set_stroke(BLUE, width=2)
        self.place_in_area(para1, 'C4', 'E6')
        self.play(FadeIn(para1))

        # === Animation for Lecture Line 3 ===
        # Display the ratio of areas: det(A1) / det(A) = x1.
        self.lecture[2].set_color(GREEN)
        ratio_text = MathTex(r"\frac{\det(A_1)}{\det(A)} = x_1").scale(0.9)
        self.place_at_grid(ratio_text, 'B4')
        self.play(Write(ratio_text))

        # === Animation for Lecture Line 4 ===
        # Animate the transition showing how area ratio corresponds to x1 coordinate.
        self.lecture[3].set_color(ORANGE)
        self.play(Indicate(para1), Indicate(ratio_text))

        # === Animation for Lecture Line 5 ===
        # Summarize the rule x_i = det(Ai) / det(A) in #FFFFFF using parallelogram icon.
        self.lecture[4].set_color(WHITE)
        para2 = get_parallelogram()
        para2.set_fill(WHITE, opacity=0.3)
        summary = MathTex(r"x_i = \frac{\det(A_i)}{\det(A)}").scale(1.2)
        summary_group = VGroup(para2, summary).arrange(DOWN)
        self.place_in_area(summary_group, 'C4', 'E6')
        self.play(ReplacementTransform(para1, para2), Write(summary))
        self.wait(2)
