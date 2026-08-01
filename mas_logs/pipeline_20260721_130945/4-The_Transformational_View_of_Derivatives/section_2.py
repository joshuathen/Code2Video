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
        # 1. Setup the standard layout
        self.setup_layout(
            "Transformational View of Derivatives",
            [
                "- Derivative as a Local Linear Map",
                "- Mapping dx to dy",
                "- Squeezing and Stretching Space",
                "- Linearization of f(x) at a Point"
            ]
        )

        # 2. Visualize Input and Output Spaces
        input_line = NumberLine(x_range=[-2, 2, 1], length=4, include_numbers=True, color=BLUE)
        output_line = NumberLine(x_range=[-2, 2, 1], length=4, include_numbers=True, color=RED)
        
        self.place_at_grid(input_line, "B3")
        self.place_at_grid(output_line, "E3")
        
        input_label = Text("Input Space (x)", font_size=18, color=BLUE).next_to(input_line, UP)
        output_label = Text("Output Space (y)", font_size=18, color=RED).next_to(output_line, UP)

        # 3. Representing the Transform (the derivative factor)
        # For f(x) = x^2 at x=1, f'(1) = 2.
        dx_arrow = Arrow(start=input_line.n2p(0), end=input_line.n2p(0.5), color=YELLOW, buff=0)
        dy_arrow = Arrow(start=output_line.n2p(0), end=output_line.n2p(1.0), color=YELLOW, buff=0)
        
        dx_text = MathTex("dx", font_size=20, color=YELLOW).next_to(dx_arrow, DOWN)
        dy_text = MathTex("dy = f'(x)dx", font_size=20, color=YELLOW).next_to(dy_arrow, DOWN)

        # 4. Animation Sequence
        self.play(Create(input_line), Write(input_label))
        self.play(Create(output_line), Write(output_label))
        self.wait(0.5)

        self.play(GrowArrow(dx_arrow), Write(dx_text))
        self.play(
            TransformFromCopy(dx_arrow, dy_arrow),
            Write(dy_text),
            run_time=2
        )
        
        # Highlight the scaling factor
        scaling_factor = Text("Scale Factor = 2", font_size=22, color=YELLOW)
        self.place_at_grid(scaling_factor, "C3")
        self.play(FadeIn(scaling_factor, shift=UP))
        
        self.wait(3)

        # Clear specific parts for next logic
        self.play(
            FadeOut(dx_arrow), FadeOut(dy_arrow), 
            FadeOut(dx_text), FadeOut(dy_text),
            FadeOut(scaling_factor)
        )
        
        # Show multiple transforms
        circles = VGroup(*[Circle(radius=0.1, color=WHITE).move_to(input_line.n2p(p)) for p in [-1, 0, 1]])
        mapped_circles = VGroup(*[Circle(radius=0.1, color=WHITE).move_to(output_line.n2p(p*2)) for p in [-1, 0, 1]])
        
        self.play(Create(circles))
        self.play(TransformFromCopy(circles, mapped_circles), run_time=2)
        
        self.wait(2)
