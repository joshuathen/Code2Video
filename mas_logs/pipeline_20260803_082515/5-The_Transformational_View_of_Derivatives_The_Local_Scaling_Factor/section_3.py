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
            "For non-linear functions, stretching is not constant.",
            "Some regions expand while others compress.",
            "Zoom in infinitely on any specific point.",
            "The transformation looks like a simple linear stretch.",
            "This is the principle of local linearity."
        ]
        self.setup_layout("The Core Concept: Local Linearity", lecture_lines)
        
        # Define line-specific colors
        c_line1 = "#E0E0E0"  # Off-white
        c_line2 = "#FF4D4D"  # Soft Red
        c_line3 = "#FFFF66"  # Light Yellow
        c_line4 = "#FF9933"  # Light Orange
        c_line5 = "#90EE90"  # Light Green
        colors = [c_line1, c_line2, c_line3, c_line4, c_line5]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        
        # Input and Output Lines
        input_line = NumberLine(x_range=[0, 2, 0.5], length=5, include_numbers=True, font_size=16, color=WHITE)
        output_line = NumberLine(x_range=[0, 4, 1], length=5, include_numbers=True, font_size=16, color=WHITE)
        
        self.place_in_area(input_line, "B1", "B6")
        self.place_in_area(output_line, "E1", "E6")
        
        input_label = Text("Input x", font_size=20, color=WHITE)
        output_label = Text("Output f(x) = x²", font_size=20, color=WHITE)
        # Resolved Issue #26: Centering labels using place_in_area
        self.place_in_area(input_label, "A1", "A6", scale_factor=1.0)
        self.place_in_area(output_label, "D1", "D6", scale_factor=1.0)

        # Mapping function: f(x) = x^2
        def f(x): return x**2
        
        # Discrete points to show non-uniform mapping
        input_pts = [0.25, 0.75, 1.25, 1.75]
        input_dots = VGroup(*[Dot(input_line.n2p(x), color=WHITE, radius=0.06) for x in input_pts])
        output_dots = VGroup(*[Dot(output_line.n2p(f(x)), color=WHITE, radius=0.06) for x in input_pts])
        
        arrows = VGroup(*[
            Arrow(
                start=input_line.n2p(x),
                end=output_line.n2p(f(x)),
                buff=0.1,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.1,
                color="#666666"
            ) for x in input_pts
        ])

        self.play(Create(input_line), Create(output_line), Write(input_label), Write(output_label))
        self.play(FadeIn(input_dots), FadeIn(output_dots), Create(arrows))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(colors[1])
        
        # Heat Map on Input Line: f'(x) = 2x
        # x=0 to 0.5: compression (f'<1), 0.5 to 2: expansion (f'>1)
        heat_line = Line(input_line.n2p(0), input_line.n2p(2), stroke_width=6)
        heat_line.set_color_by_gradient("#0000FF", "#FFFFFF", "#FF0000") # Blue (compress) -> Red (expand)
        
        comp_label = Text("Compress", font_size=14, color="#AAAAFF")
        exp_label = Text("Expand", font_size=14, color="#FFAAAA")
        self.place_at_grid(comp_label, "C1", scale_factor=0.9)
        self.place_at_grid(exp_label, "C6", scale_factor=0.9)
        
        self.play(Create(heat_line), FadeIn(comp_label), FadeIn(exp_label))
        self.play(arrows.animate.set_color(colors[1]))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(colors[2])
        
        # Choose specific point 'a'
        a_val = 1.25
        a_point = Dot(input_line.n2p(a_val), color=colors[2], radius=0.08)
        a_text = MathTex("a", color=colors[2], font_size=24).next_to(a_point, UP, buff=0.1)
        
        # Visual magnifying glass
        magnifier = Circle(radius=0.4, color=colors[2], stroke_width=3).move_to(input_line.n2p(a_val))
        
        self.play(FadeIn(a_point), Write(a_text), Create(magnifier))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(colors[3])
        
        # Zoom into point 'a'. Define a very small local range.
        z_range = [1.15, 1.35]
        z_input_line = NumberLine(x_range=z_range, length=5, color=WHITE, include_numbers=False)
        z_output_line = NumberLine(x_range=[f(z_range[0]), f(z_range[1])], length=5, color=WHITE, include_numbers=False)
        
        # Reposition zoomed lines in the same general areas
        self.place_in_area(z_input_line, "B1", "B6")
        self.place_in_area(z_output_line, "E1", "E6")
        
        # Local dots will look uniformly spaced
        z_pts = np.linspace(z_range[0], z_range[1], 5)
        z_in_dots = VGroup(*[Dot(z_input_line.n2p(x), color=WHITE, radius=0.06) for x in z_pts])
        z_out_dots = VGroup(*[Dot(z_output_line.n2p(f(x)), color=WHITE, radius=0.06) for x in z_pts])
        
        z_arrows = VGroup(*[
            Arrow(
                start=z_input_line.n2p(x),
                end=z_output_line.n2p(f(x)),
                buff=0.1,
                stroke_width=2,
                max_tip_length_to_length_ratio=0.1,
                color=colors[3]
            ) for x in z_pts
        ])

        zoom_note = Text("Zoomed View (Local Scale)", font_size=18, color=colors[3])
        # Resolved Issue #27: Centering zoom_note using place_in_area
        self.place_in_area(zoom_note, "A1", "A6", scale_factor=1.0)

        # Simulation of "zooming in" by fading out macro view and fading in micro view
        self.play(
            FadeOut(input_line), FadeOut(output_line), FadeOut(input_dots), FadeOut(output_dots),
            FadeOut(arrows), FadeOut(heat_line), FadeOut(comp_label), FadeOut(exp_label),
            FadeOut(magnifier), FadeOut(a_point), FadeOut(a_text),
            FadeIn(z_input_line), FadeIn(z_output_line), FadeIn(z_in_dots), FadeIn(z_out_dots), FadeIn(z_arrows),
            Write(zoom_note)
        )
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(colors[4])
        
        # Highlight the local linearity (the arrows are roughly the same length and slope)
        highlight = SurroundingRectangle(z_arrows, color=colors[4], buff=0.2)
        local_lin_label = Text("Locally Linear Segment", color=colors[4], font_size=20)
        # Resolved Issue #28: Centering local_lin_label using place_in_area
        self.place_in_area(local_lin_label, "C1", "C6", scale_factor=1.0)
        
        self.play(Create(highlight), Write(local_lin_label))
        self.wait(2)
