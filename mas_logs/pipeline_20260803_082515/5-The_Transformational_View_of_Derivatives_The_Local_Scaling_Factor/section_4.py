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
        # === Setup ===
        title = "Defining the Derivative as a Scaling Factor"
        lines = [
            "The derivative f prime of x is a ratio.",
            "It represents the local scaling factor at a point.",
            "Values greater than one indicate local expansion.",
            "Values between zero and one indicate local compression.",
            "Negative values mean the space is being flipped."
        ]
        self.setup_layout(title, lines)
        
        COLOR_FORMULA = WHITE
        COLOR_INPUT = BLUE_C
        COLOR_OUTPUT = GREEN_C
        COLOR_HIGHLIGHT = "#FFFF00" # Yellow as requested
        COLOR_NEGATIVE = RED_C

        # === Animation for Lecture Line 1 ===
        # Display the derivative formula f'(x) = dy/dx in #FFFFFF.
        self.play(self.lecture[0].animate.set_color(COLOR_FORMULA))
        
        formula = MathTex(r"f'(x) = \frac{dy}{dx}", color=COLOR_FORMULA)
        self.place_in_area(formula, "A3", "A4", scale_factor=1.2)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # At x=3, show a tiny increment dx = 0.001 on the input line.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_INPUT)
        )

        input_line = NumberLine(x_range=[0, 5, 1], length=5, include_numbers=True, color=COLOR_INPUT, font_size=18)
        self.place_in_area(input_line, "C1", "C6")
        input_label = Text("Input", font_size=18, color=COLOR_INPUT)
        # Issue #29 fix: Move input_label to B1 to avoid overlap with axis
        self.place_at_grid(input_label, 'B1', scale_factor=0.8)

        output_line = NumberLine(x_range=[0, 30, 10], length=5, include_numbers=True, color=COLOR_OUTPUT, font_size=18)
        self.place_in_area(output_line, "E1", "E6")
        output_label = Text("Output", font_size=18, color=COLOR_OUTPUT)
        # Issue #30 fix: Move output_label to D1 to avoid overlap with axis
        self.place_at_grid(output_label, 'D1', scale_factor=0.8)

        self.play(Create(input_line), Create(output_line), Write(input_label), Write(output_label))

        # Mark x=3
        x_point = input_line.n2p(3)
        dot_x = Dot(x_point, color=WHITE, radius=0.05)
        label_x = MathTex("x=3", font_size=20, color=WHITE).next_to(dot_x, UP, buff=0.1)
        
        # Visual dx (magnified for clarity but labeled correctly)
        dx_val = 0.5 
        dx_line = Line(input_line.n2p(3), input_line.n2p(3 + dx_val), color=COLOR_HIGHLIGHT, stroke_width=6)
        dx_text = MathTex("dx = 0.001", font_size=20, color=COLOR_HIGHLIGHT).next_to(dx_line, DOWN, buff=0.1)

        self.play(FadeIn(dot_x), Write(label_x))
        self.play(Create(dx_line), Write(dx_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Values greater than one indicate local expansion.
        # Show the resulting change dy = 0.006 on the output line, highlighting the 6x expansion.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )

        # Mapping: dx=0.5 visual on input [0,5] -> dy=3.0 visual on output [0,30]
        # Label says dy=0.006 (which is 6 * 0.001)
        dy_start = output_line.n2p(10) # Centering around 10 for visibility
        dy_end = output_line.n2p(10 + 6) # dy length = 6 in output scale
        dy_line = Line(dy_start, dy_end, color=COLOR_HIGHLIGHT, stroke_width=6)
        dy_text = MathTex("dy = 0.006", font_size=20, color=COLOR_HIGHLIGHT).next_to(dy_line, DOWN, buff=0.1)
        
        # Connecting arrows to show mapping
        arrow1 = Arrow(input_line.n2p(3), dy_start, color=GRAY_C, buff=0.1, stroke_width=2, max_tip_length_to_length_ratio=0.1)
        arrow2 = Arrow(input_line.n2p(3+dx_val), dy_end, color=GRAY_C, buff=0.1, stroke_width=2, max_tip_length_to_length_ratio=0.1)

        self.play(Create(arrow1), Create(arrow2))
        self.play(Create(dy_line), Write(dy_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Values between zero and one indicate local compression.
        # Label the local scaling factor as '6' in #FFFF00 near the output line.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        scaling_label = Text("Scale Factor: 6", font_size=24, color=COLOR_HIGHLIGHT)
        self.place_at_grid(scaling_label, "D5")
        self.play(Write(scaling_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Negative values mean the space is being flipped.
        # Show a mapping where f'(x) is negative, causing the output arrows to flip or cross.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_NEGATIVE)
        )

        # Transition to negative case - Cleanup
        self.play(
            FadeOut(dx_line, dx_text, dy_line, dy_text, arrow1, arrow2, scaling_label, dot_x, label_x)
        )

        # Crossing arrows to indicate negative derivative (flipping)
        neg_arrow1 = Arrow(input_line.n2p(1), output_line.n2p(25), color=COLOR_NEGATIVE, buff=0.1, stroke_width=3)
        neg_arrow2 = Arrow(input_line.n2p(4), output_line.n2p(5), color=COLOR_NEGATIVE, buff=0.1, stroke_width=3)
        
        flip_text = Text("Negative Scaling = Flipping", font_size=20, color=COLOR_NEGATIVE)
        # Issue #31 fix: use place_in_area to avoid overflow/overlap
        self.place_in_area(flip_text, 'D2', 'D4')

        self.play(Create(neg_arrow1), Create(neg_arrow2), Write(flip_text))
        self.wait(2)
