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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Defining the Derivative as an Operator", [
            "Derivative is a slope-machine.",
            "Input curve, output slope function.",
            "Maps input to rate.",
            "Instantaneous rate of change.",
            "A dynamic new function."
        ])
        
        # Elements
        machine = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg", color=WHITE)
        self.place_in_area(machine, "B2", "C3", scale_factor=0.6)
        label_f = MathTex("f(x)", color=WHITE).next_to(machine, LEFT)
        label_df = MathTex("f'(x)", color=WHITE).next_to(machine, RIGHT)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.play(Create(machine), Write(label_f), Write(label_df))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(ORANGE)
        curve = FunctionGraph(lambda x: 0.1 * x**3, color=ORANGE, x_range=[-2, 2])
        slope_curve = FunctionGraph(lambda x: 0.3 * x**2, color=GREEN, x_range=[-2, 2])
        
        # Using machine position
        self.play(
            FadeIn(curve),
            curve.animate.move_to(machine.get_left()),
            run_time=2
        )
        self.play(
            FadeIn(slope_curve),
            slope_curve.animate.move_to(machine.get_right()),
            run_time=2
        )

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        dot = Dot(color=GREEN)
        self.place_at_grid(dot, "D3", scale_factor=0.7)
        self.play(FadeIn(dot))
        self.play(dot.animate.shift(RIGHT * 2))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW)
        formula = MathTex(r"\\frac{d}{dx} [f(x)] = f'(x)", color=YELLOW)
        self.place_at_grid(formula, "B5", scale_factor=0.6)
        self.play(Flash(formula, color=YELLOW))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        # Using the machine for the point mapping animation requested in storyboard
        self.play(Indicate(machine))
