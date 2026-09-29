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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = ["A vector as a linear machine.", "Covectors output scalar scores.", "Dot product powers this mechanism."]
        self.setup_layout("Defining Duality: The Linear Functional", lecture_lines)
        
        # Assets
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        score_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scorecard.svg")
        
        # === Animation for Lecture Line 1 ===
        # Display linear functional f(x) = c^T x in #FFFFFF
        formula = MathTex(r"f(\mathbf{x}) = \mathbf{c}^T \mathbf{x}", color=WHITE)
        self.place_in_area(formula, 'B3', 'C5', scale_factor=1.0)
        self.place_at_grid(calc_icon, "B2", scale_factor=0.6)
        self.play(Write(formula), FadeIn(calc_icon))
        self.lecture[0].set_color(ORANGE)

        # === Animation for Lecture Line 2 ===
        # Draw a scalar mapping as a line segment in #00CED1
        scalar_line = Line(start=np.array([-1, 0, 0]), end=np.array([1, 0, 0]), color="#00CED1")
        self.place_at_grid(scalar_line, "D3", scale_factor=1.0)
        scalar_label = Text("Scalar Output", font_size=18, color="#00CED1")
        self.place_at_grid(scalar_label, "E4", scale_factor=0.8)
        self.play(Create(scalar_line), Write(scalar_label))
        self.lecture[1].set_color("#00CED1")

        # === Animation for Lecture Line 3 ===
        # Animate input vector x being transformed to scalar f(x) and update value on scorecard
        vector_x = Arrow(start=ORIGIN, end=UP*1.5, color=YELLOW)
        self.place_at_grid(vector_x, "C2", scale_factor=0.8)
        
        dot = Dot(color=WHITE)
        self.place_at_grid(dot, "D4", scale_factor=0.8)
        self.place_at_grid(score_icon, "E5", scale_factor=0.6)
        
        self.play(GrowArrow(vector_x))
        self.play(vector_x.animate.move_to(self.grid["D3"]), run_time=1.5)
        self.play(FadeOut(vector_x), FadeIn(dot), FadeIn(score_icon))
        self.lecture[2].set_color(YELLOW)
        
        self.wait(2)
