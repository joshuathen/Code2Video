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
        lecture_lines = [
            "Formulate the Newton-Raphson iterative step.",
            "x_{n+1} equals x_n minus f/f'.",
            "Refine estimates with each iteration.",
            "Starts must be close enough.",
            "Automatically converge on the solution."
        ]
        self.setup_layout("The Newton-Raphson Iteration", lecture_lines)
        
        # Assets
        formula = MathTex("x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}", color=WHITE)
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        ruler_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        compass_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        graph_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/graph.svg")
        exclamation_mark = Text("!", color=RED, font_size=40)
        
        # Layout according to requirements
        self.place_in_area(formula, 'B2', 'D5', scale_factor=1.2)
        self.place_at_grid(calc_icon, 'A4', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(formula), FadeIn(calc_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(Indicate(formula))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.place_at_grid(ruler_icon, 'D4', scale_factor=0.6)
        self.play(FadeIn(ruler_icon))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(RED))
        self.place_at_grid(compass_icon, 'E4', scale_factor=0.6)
        self.place_at_grid(exclamation_mark, 'D6', scale_factor=0.6)
        self.play(FadeIn(compass_icon), FadeIn(exclamation_mark))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(GREEN))
        self.place_at_grid(graph_icon, 'F4', scale_factor=0.6)
        self.play(FadeIn(graph_icon))
        self.wait(1)
