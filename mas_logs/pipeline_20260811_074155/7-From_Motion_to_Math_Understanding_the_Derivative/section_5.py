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
        # Data for setup
        title = "Defining the Derivative"
        lines = [
            "We define the derivative as this specific limit.",
            "It calculates the slope at any single point.",
            "Let's apply this to a simple quadratic function.",
            "The derivative formula reveals the exact instantaneous speed.",
            "Calculus turns motion into precise mathematical equations."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_LINE_1 = WHITE
        COLOR_LINE_2 = "#58C4DD" # Blue
        COLOR_LINE_3 = "#F4D03F" # Yellow
        COLOR_LINE_4 = "#E67E22" # Orange
        COLOR_LINE_5 = "#90EE90" # Light Green

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_LINE_1)
        
        # Formal Definition Formula
        formula = MathTex(
            "f'(x) = \\lim_{h \\to 0} \\frac{f(x+h) - f(x)}{h}",
            color=WHITE,
            font_size=36
        )
        formula_box = SurroundingRectangle(formula, color=WHITE, buff=0.2)
        formula_group = VGroup(formula_box, formula)
        # Fix: Issue 33 - Use A2-B5 area for better focus
        self.place_in_area(formula_group, 'A2', 'B5', scale_factor=0.9)
        
        self.play(Create(formula_box), Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_LINE_2)
        
        # Highlight f'(x) and explain it represents slope
        slope_highlight = SurroundingRectangle(formula[0][0:5], color=COLOR_LINE_2, buff=0.1)
        self.play(Create(slope_highlight))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_LINE_3)
        
        # Function Machine
        machine_rect = RoundedRectangle(corner_radius=0.2, height=1.5, width=2.5, color=COLOR_LINE_3)
        machine_label = Text("Derivative Machine", font_size=18, color=COLOR_LINE_3)
        machine = VGroup(machine_rect, machine_label)
        # Fix: Issue 34 - Position machine at C3-D5
        self.place_in_area(machine, 'C3', 'D5', scale_factor=1.0)
        
        input_func = MathTex("f(x) = x^2", color=COLOR_LINE_3, font_size=24)
        output_func = MathTex("f'(x) = 2x", color=COLOR_LINE_3, font_size=24)
        
        # Fix: Issue 34 - Position input at C2
        self.place_at_grid(input_func, 'C2', scale_factor=1.0)
        self.place_at_grid(output_func, 'D6', scale_factor=1.0)
        
        input_arrow = Arrow(input_func.get_right(), machine_rect.get_left(), color=COLOR_LINE_3, buff=0.1)
        output_arrow = Arrow(machine_rect.get_right(), output_func.get_left(), color=COLOR_LINE_3, buff=0.1)
        
        self.play(FadeIn(machine), FadeIn(input_func), Create(input_arrow))
        self.wait(0.5)
        self.play(FadeIn(output_func), Create(output_arrow))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_LINE_4)
        
        # Asset: Cheetah icon
        # Issue 21 - Integrate cheetah SVG
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        self.place_at_grid(cheetah, 'E1', scale_factor=0.5)
        
        # Calculation at t=3
        calc_text = MathTex(
            "f'(3) = 2(3) = 6 \\text{ m/s}",
            color=COLOR_LINE_4,
            font_size=30
        )
        # Fix: Issue 35 - Use E3-E5 for calc_text
        self.place_in_area(calc_text, 'E3', 'E5', scale_factor=1.0)
        
        t_label = Text("at t=3", font_size=18, color=COLOR_LINE_4)
        # Fix: Issue 35 - Position t_label at E2
        self.place_at_grid(t_label, 'E2', scale_factor=1.0)
        
        self.play(FadeIn(cheetah), Write(t_label), Write(calc_text))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_LINE_5)
        
        # Final Label
        final_label = Text("The Derivative", color=COLOR_LINE_5, weight=BOLD, font_size=32)
        self.place_at_grid(final_label, 'F3', scale_factor=1.0)
        
        self.play(Write(final_label))
        self.play(final_label.animate.scale(1.1).set_color(YELLOW))
        self.wait(2)
