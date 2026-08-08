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
            "For non-linear functions, the scaling varies at every point.",
            "Imagine an ant walking along the input number line.",
            "At x equals three, the ground stretches six times wider.",
            "Zooming in, the transformation looks like a uniform stretch.",
            "The derivative captures this local, linear expansion factor."
        ]
        self.setup_layout("The Elastic Ant: Visualizing f(x) = x²", lecture_lines)

        # Colors for matching lecture lines
        COLOR_FUNC = "#FFFFFF" # White
        COLOR_ANT = "#FFFF00"  # Yellow
        COLOR_INPUT = "#FF0000" # Red
        COLOR_OUTPUT = "#00FF00" # Green
        COLOR_ZOOM = "#00FFFF"  # Cyan
        COLOR_DERIV = "#FFC0CB" # Pink

        # === Animation for Lecture Line 1 ===
        # "For non-linear functions, the scaling varies at every point."
        self.play(self.lecture[0].animate.set_color(COLOR_FUNC))
        
        func_text = MathTex("f(x) = x^2", color=COLOR_FUNC)
        self.place_in_area(func_text, "A2", "A5", scale_factor=1.2)
        
        # Issue 32: Move lines to column 2 to avoid lecture text
        input_line = NumberLine(x_range=[0, 4, 1], length=4, include_numbers=True, font_size=16)
        output_line = NumberLine(x_range=[0, 10, 2], length=4, include_numbers=True, font_size=16)
        
        self.place_in_area(input_line, "B2", "B6")
        self.place_in_area(output_line, "E2", "E6")
        
        input_label = Text("Input (x)", font_size=16, color=BLUE_B)
        output_label = Text("Output (f(x))", font_size=16, color=GREEN_B)
        
        # Issue 32: Position labels at B2 and E2
        self.place_at_grid(input_label, "B2", scale_factor=0.8).shift(UP * 0.4)
        self.place_at_grid(output_label, "E2", scale_factor=0.8).shift(UP * 0.4)

        self.play(
            Write(func_text),
            Create(input_line),
            Create(output_line),
            Write(input_label),
            Write(output_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Imagine an ant walking along the input number line."
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(COLOR_ANT)
        )
        
        # Issue 25: Use SVG ant asset
        ant = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ant.svg")
        ant.set_color(COLOR_ANT)
        ant.scale(0.15)
        # Position at x=2 initially
        ant.move_to(input_line.n2p(2) + UP * 0.2)
        
        self.play(FadeIn(ant))
        # Ant walks to x=3.
        self.play(ant.animate.move_to(input_line.n2p(3) + UP * 0.2), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "At x equals three, the ground stretches six times wider."
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(COLOR_INPUT)
        )
        
        # Highlight segment on input line around x=3 (0.2 units: 2.9 to 3.1)
        input_segment = Line(input_line.n2p(2.9), input_line.n2p(3.1), color=COLOR_INPUT, stroke_width=8)
        
        # Highlight segment on output line around f(3)=9
        # f(2.9)=8.41, f(3.1)=9.61. Length = 1.2 (which is 6 * 0.2)
        output_segment = Line(output_line.n2p(8.41), output_line.n2p(9.61), color=COLOR_OUTPUT, stroke_width=8)
        
        # Issue 34: Move stretch_label to D4
        stretch_label = MathTex("\\text{Stretch: } 6\\times", color=COLOR_INPUT)
        self.place_at_grid(stretch_label, "D4", scale_factor=0.8)
        
        # Visual mapping connection
        mapping_arrow = Arrow(start=input_line.n2p(3), end=output_line.n2p(9), color=WHITE, stroke_width=2, buff=0.2)

        self.play(
            Create(input_segment),
            Create(output_segment),
            Write(stretch_label),
            GrowArrow(mapping_arrow)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Zooming in, the transformation looks like a uniform stretch."
        self.play(
            self.lecture[2].animate.set_color(GRAY),
            self.lecture[3].animate.set_color(COLOR_ZOOM)
        )
        
        # Zoom effect focused on x=3 and f(3)=9
        zoom_factor = 4
        self.play(
            input_line.animate.scale(zoom_factor, about_point=input_line.n2p(3)),
            output_line.animate.scale(zoom_factor, about_point=output_line.n2p(9)),
            input_segment.animate.scale(zoom_factor, about_point=input_line.n2p(3)),
            output_segment.animate.scale(zoom_factor, about_point=output_line.n2p(9)),
            ant.animate.scale(2, about_point=input_line.n2p(3)),
            FadeOut(stretch_label),
            FadeOut(mapping_arrow),
            FadeOut(input_label),
            FadeOut(output_label),
            FadeOut(func_text)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "The derivative captures this local, linear expansion factor."
        self.play(
            self.lecture[3].animate.set_color(GRAY),
            self.lecture[4].animate.set_color(COLOR_DERIV)
        )
        
        deriv_formula = MathTex("f'(x) = 2x", color=COLOR_DERIV)
        deriv_at_3 = MathTex("f'(3) = 6", color=COLOR_DERIV)
        
        # Issue 33: Move derivative formulas to D2 and D5
        self.place_at_grid(deriv_formula, "D2", scale_factor=1.0)
        self.place_at_grid(deriv_at_3, "D5", scale_factor=1.0)
        
        self.play(Write(deriv_formula))
        self.wait(0.5)
        self.play(Write(deriv_at_3))
        
        self.wait(2)
