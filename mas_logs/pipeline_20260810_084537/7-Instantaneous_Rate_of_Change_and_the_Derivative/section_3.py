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
        self.setup_layout("Defining the Derivative via Limits", [
            "Derivative is limit as interval shrinks.",
            "Formally, f prime of x limit notation.",
            "Slope represents instantaneous rate of change.",
            "Ball path shows changing tangent slope.",
            "Slope is positive, zero, then negative."
        ])
        
        # Load asset
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        
        # Definition
        formula = MathTex(r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}")
        # Note: indices depend on the specific string formatting in LaTeX
        # f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
        # In current MathTex structure this might require manual mapping
        diff_quotient = formula[0][9:23] # numerator
        limit_h = formula[0][5:9] 
        
        # Applying requested position adjustment from VideoCritic:
        # B3 to D5, scale 1.2
        self.place_in_area(formula, 'B3', 'D5', scale_factor=1.2)
        
        # Animation for Lecture Line 1
        self.play(Write(formula))
        self.lecture[0].set_color(YELLOW)
        self.place_at_grid(ball.copy(), 'A6', scale_factor=0.5)

        # Animation for Lecture Line 2
        self.play(Indicate(diff_quotient, color="#FF00FF"), run_time=1.5)
        self.lecture[1].set_color("#FF00FF")

        # Animation for Lecture Line 3
        self.play(Indicate(limit_h, color="#00FFFF"), run_time=1.5)
        self.lecture[2].set_color("#00FFFF")

        # Animation for Lecture Line 4
        dot = Dot(color=WHITE)
        self.place_at_grid(dot, 'E4', scale_factor=1.0)
        self.play(FadeIn(dot))
        self.play(dot.animate.shift(UP*1.5), run_time=2)
        self.lecture[3].set_color(WHITE)

        # Animation for Lecture Line 5
        box = SurroundingRectangle(formula, color="#00FF00", buff=0.1)
        self.play(Create(box))
        self.lecture[4].set_color("#00FF00")
        
        # Adding ball as requested in storyboard
        self.place_at_grid(ball, 'F6', scale_factor=0.5)
        
        self.wait(2)
