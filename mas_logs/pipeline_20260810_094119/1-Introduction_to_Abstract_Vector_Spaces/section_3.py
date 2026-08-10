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
        lecture_lines = [
            "Polynomials behave just like geometric vectors.",
            "Functions also follow vector space rules.",
            "Zero polynomial acts as our zero.",
            "Algebraic structure defines both categories.",
            "Abstraction links disparate mathematical objects."
        ]
        self.setup_layout("Case Study: Beyond Geometric Arrows", lecture_lines)
        
        # Elements
        # Using SVG for the arrow as per storyboard asset request
        arrow_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        poly_graph = FunctionGraph(lambda x: 0.5 * x**2 - 1, color="#FF9900", x_range=[-2, 2])
        poly_label = MathTex("P(x) = ax^2 + bx + c", color="#FF9900")
        coeff_vec = MathTex(r"\vec{v} = \begin{bmatrix} a \\ b \\ c \end{bmatrix}", color="#99FF00")
        zero_poly = MathTex("0(x) = 0", color=BLUE)
        
        poly_group = VGroup(poly_graph, poly_label).arrange(DOWN)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF9900"))
        self.place_at_grid(arrow_icon, "B1", scale_factor=0.5)
        self.place_in_area(poly_group, "B2", "C5", scale_factor=0.6)
        self.play(Create(arrow_icon), Create(poly_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#99FF00"))
        self.place_at_grid(coeff_vec, "D3", scale_factor=0.7)
        self.play(Write(coeff_vec))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        self.place_at_grid(zero_poly, "D5", scale_factor=0.8)
        self.play(FadeIn(zero_poly))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(PURPLE))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(RED))
        self.play(Indicate(poly_group), Indicate(coeff_vec))
        self.wait(2)
