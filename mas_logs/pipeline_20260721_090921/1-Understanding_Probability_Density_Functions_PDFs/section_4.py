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
        # Setup the title and lecture lines
        title = "Calculating Probability: The Integral Slice"
        lines = [
            "Integration calculates probability for a specific range.",
            "We shade a slice under the curve between points.",
            "Shading from a to b gives the interval's probability."
        ]
        self.setup_layout(title, lines)

        # Colors and constants
        PDF_COLOR = "#00FFFF" # Cyan
        SHADE_COLOR = "#FF0000" # Red
        HIGHLIGHT_COLOR = "#FFFF00" # Yellow
        
        # Create Axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[0, 0.5, 0.1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": False, "color": WHITE}
        )
        
        def pdf_func(x):
            # Standard normal distribution formula
            return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

        curve = axes.plot(pdf_func, color=PDF_COLOR)
        
        # Group and position the plot
        axes_group = VGroup(axes, curve)
        # Issue 29: Position axes_group in C1-F6 to leave room for formula and avoid column 1
        self.place_in_area(axes_group, 'C1', 'F6', scale_factor=0.85)
        
        # Vertical markers for interval [a, b]
        a_val, b_val = -1.0, 1.2
        v_line_a = axes.get_vertical_line(axes.input_to_graph_point(a_val, curve), color=WHITE, stroke_width=2)
        v_line_b = axes.get_vertical_line(axes.input_to_graph_point(b_val, curve), color=WHITE, stroke_width=2)
        
        # Labels for a and b - positioned relative to moved axes
        a_label = Text("a", color=WHITE, font_size=24).next_to(axes.c2p(a_val, 0), DOWN, buff=0.1)
        b_label = Text("b", color=WHITE, font_size=24).next_to(axes.c2p(b_val, 0), DOWN, buff=0.1)

        # Formula text
        formula = Text("P(a < X < b) = Area", color=WHITE, font_size=32)
        # Issue 28: Position formula in B2-B5 for horizontal centering and avoiding title area
        self.place_in_area(formula, 'B2', 'B5', scale_factor=1.0)

        # Shaded area
        area = axes.get_area(curve, x_range=[a_val, b_val], color=SHADE_COLOR, opacity=0.5)

        # === Animation for Lecture Line 1 ===
        # "Integration calculates probability for a specific range."
        # Highlight line 1 with curve color (Cyan) to match animation elements
        self.play(self.lecture[0].animate.set_color(PDF_COLOR))
        self.play(Create(axes), Create(curve), run_time=1.5)
        self.play(Create(v_line_a), Create(v_line_b), Write(a_label), Write(b_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We shade a slice under the curve between points."
        # Highlight line 2 with shading color (Red) to match animation elements
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(SHADE_COLOR)
        )
        self.play(FadeIn(area, scale=0.9), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Shading from a to b gives the interval's probability."
        # Highlight line 3 with general highlight color (Yellow)
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        self.play(Write(formula))
        self.wait(3)

        # Reset final line color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
