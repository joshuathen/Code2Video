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
        title = "The Local Transformation: Zooming In"
        lines = [
            "Non-linear functions don't stretch space uniformly.",
            "The derivative represents scaling at a specific point.",
            "Zoom in until the transformation looks like linear scaling.",
            "Tiny input nudge dx yields tiny output nudge df.",
            "The ratio df/dx is the local magnification factor."
        ]
        self.setup_layout(title, lines)

        # Colors
        color1 = "#E0FFFF"
        color2 = "#ADD8E6"
        color3 = "#87CEFA"
        color4 = "#00BFFF"
        color5 = "#1E90FF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color1)
        func_label = MathTex("f(x) = x^2", color=color1)
        self.place_at_grid(func_label, "A1", scale_factor=0.8)
        
        # Create Number Lines
        input_line = NumberLine(x_range=[0, 6, 1], length=5, include_numbers=True, color=WHITE).scale(0.8)
        output_line = NumberLine(x_range=[0, 40, 10], length=5, include_numbers=True, color=WHITE).scale(0.8)
        
        self.place_in_area(input_line, "B1", "B6")
        self.place_in_area(output_line, "E1", "E6")
        
        input_title = Text("Input Space", font_size=16, color=WHITE)
        output_title = Text("Output Space", font_size=16, color=WHITE)
        # Fix for Issue 27: Reposition titles
        self.place_in_area(input_title, "A5", "A6", scale_factor=0.7)
        self.place_in_area(output_title, "D5", "D6", scale_factor=0.7)

        # Focus point x=3
        dot_x = Dot(input_line.n2p(3), color=color1)
        label_x = MathTex("x=3", font_size=20, color=color1).next_to(dot_x, UP, buff=0.1)
        
        dot_fx = Dot(output_line.n2p(9), color=color1)
        label_fx = MathTex("f(3)=9", font_size=20, color=color1).next_to(dot_fx, DOWN, buff=0.1)

        self.play(
            Write(func_label),
            Create(input_line),
            Create(output_line),
            Write(input_title),
            Write(output_title),
            FadeIn(dot_x, label_x),
            FadeIn(dot_fx, label_fx)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(color2)
        
        derivative_label = MathTex("f'(x) = 2x", color=color2)
        # Fix for Issue 26: Relocate labels
        self.place_at_grid(derivative_label, "A2", scale_factor=0.8)
        
        derivative_val = MathTex("f'(3) = 6", color=color2)
        # Fix for Issue 26: Relocate labels
        self.place_at_grid(derivative_val, "A3", scale_factor=0.8)

        self.play(Write(derivative_label))
        self.play(Write(derivative_val))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(color3)

        # Zoom in by replacing the lines with closer views
        input_line_zoomed = NumberLine(x_range=[2.8, 3.2, 0.1], length=5, include_numbers=True, color=WHITE).scale(0.8)
        output_line_zoomed = NumberLine(x_range=[7.8, 10.2, 0.5], length=5, include_numbers=True, color=WHITE).scale(0.8)
        
        self.place_in_area(input_line_zoomed, "B1", "B6")
        self.place_in_area(output_line_zoomed, "E1", "E6")
        
        dot_x_zoomed = Dot(input_line_zoomed.n2p(3), color=color3)
        dot_fx_zoomed = Dot(output_line_zoomed.n2p(9), color=color3)

        self.play(
            ReplacementTransform(input_line, input_line_zoomed),
            ReplacementTransform(output_line, output_line_zoomed),
            ReplacementTransform(dot_x, dot_x_zoomed),
            ReplacementTransform(dot_fx, dot_fx_zoomed),
            FadeOut(label_x, label_fx)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(color4)

        # Tiny nudge dx = 0.05
        dx_val = 0.05
        dx_start = 3
        dx_end = 3 + dx_val
        
        # Segment for dx
        dx_line = Line(
            input_line_zoomed.n2p(dx_start), 
            input_line_zoomed.n2p(dx_end), 
            color=color4, stroke_width=8
        )
        dx_brace = Brace(dx_line, UP, buff=0.05)
        dx_label = MathTex("dx", font_size=20, color=color4).next_to(dx_brace, UP, buff=0.05)
        
        # Segment for df = f'(3) * dx = 6 * 0.05 = 0.3
        df_val = 0.3
        df_start = 9
        df_end = 9 + df_val
        
        df_line = Line(
            output_line_zoomed.n2p(df_start), 
            output_line_zoomed.n2p(df_end), 
            color=color4, stroke_width=8
        )
        df_brace = Brace(df_line, DOWN, buff=0.05)
        df_label = MathTex("df", font_size=20, color=color4).next_to(df_brace, DOWN, buff=0.05)

        # Mapping arrow
        arrow = Arrow(
            input_line_zoomed.n2p(3.025),
            output_line_zoomed.n2p(9.15),
            color=color4, buff=0.1
        )

        self.play(Create(dx_line), FadeIn(dx_brace, dx_label))
        self.play(GrowArrow(arrow))
        self.play(Create(df_line), FadeIn(df_brace, df_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(color5)
        
        # Issue 20: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifier.svg]
        magnifier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifier.svg")
        self.place_at_grid(magnifier, "C3", scale_factor=0.5)
        
        ratio_formula = MathTex(
            "\\frac{df}{dx} = \\frac{0.3}{0.05} = 6", 
            color=color5
        )
        # Fix for Issue 28: Reposition ratio formula and final label
        self.place_in_area(ratio_formula, "F2", "F4", scale_factor=0.8)
        
        final_label = MathTex("f'(3) = 6", color=color5)
        # Fix for Issue 28: Reposition final label
        self.place_at_grid(final_label, "F5", scale_factor=0.8)

        self.play(FadeIn(magnifier))
        self.play(Write(ratio_formula))
        self.play(FadeIn(final_label))
        self.wait(2)
