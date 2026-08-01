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
            "For continuous variables, we transition from sums to integrals.",
            "The probability density of the sum is the convolution.",
            "The formula integrates the product of the two densities.",
            "The term z minus x is the remaining value.",
            "This calculates the probability density at any point z."
        ]
        self.setup_layout("The Continuous Leap: The Convolution Formula", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # For continuous variables, we transition from sums to integrals.
        self.lecture[0].set_color(YELLOW)
        sum_sym = MathTex(r"\sum", font_size=80, color=YELLOW)
        int_sym = MathTex(r"\int", font_size=80, color=YELLOW)
        
        # Issue 48: Move 'sum_sym' to 'B2' (scale 1.2)
        self.place_at_grid(sum_sym, "B2", scale_factor=1.2)
        self.play(FadeIn(sum_sym))
        self.wait(1)
        
        # Transform with pulse effect
        self.play(
            Transform(sum_sym, int_sym.move_to(sum_sym)),
            Flash(sum_sym, color=YELLOW, flash_radius=0.5)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The probability density of the sum is the convolution.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        formula = MathTex(
            r"f_Z(z) = \int_{-\infty}^{\infty} f_X(x) f_Y(z-x) dx",
            font_size=40,
            color=WHITE
        )
        # Issue 48: Place 'formula' in area 'B2' to 'C6' (scale 0.8)
        self.place_in_area(formula, "B2", "C6", scale_factor=0.8)
        
        self.play(
            FadeOut(sum_sym),
            Write(formula)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The formula integrates the product of the two densities.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Axes for PDF curves
        # Issue 48: Move 'axes' to area 'D2' to 'F6' (scale 0.8)
        axes = Axes(
            x_range=[-3, 4, 1],
            y_range=[0, 1.2, 0.5],
            axis_config={"include_tip": False, "font_size": 20},
            x_length=4.5,
            y_length=2.5
        )
        self.place_in_area(axes, "D2", "F6", scale_factor=0.8)
        
        # Define two PDFs
        # f_X is centered at 0
        def pdf_x_func(x): return np.exp(-x**2)
        # f_Y(z-x) is centered at x=z. Let's pick z=1.5
        def pdf_y_shifted_func(x): return 0.8 * np.exp(-(x-1.5)**2)
        
        curve_x = axes.plot(pdf_x_func, color="#ADD8E6")
        curve_y = axes.plot(pdf_y_shifted_func, color="#90EE90")
        
        label_x = MathTex("f_X(x)", color="#ADD8E6", font_size=20)
        label_y = MathTex("f_Y(z-x)", color="#90EE90", font_size=20)
        
        # Positioning labels near curves relative to axes
        label_x.move_to(axes.c2p(-1.8, 0.8))
        label_y.move_to(axes.c2p(2.8, 0.6))

        self.play(Create(axes), Create(curve_x), Create(curve_y))
        self.play(FadeIn(label_x), FadeIn(label_y))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The term z minus x is the remaining value.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Mark point x on f_X and z-x on f_Y
        x_val = 0.5
        
        point_x = axes.input_to_graph_point(x_val, curve_x)
        line_x = DashedLine(start=axes.c2p(x_val, 0), end=point_x, color=WHITE)
        dot_x = Dot(point_x, color=WHITE, radius=0.06)
        
        point_y = axes.input_to_graph_point(x_val, curve_y)
        line_y = DashedLine(start=axes.c2p(x_val, 0), end=point_y, color=WHITE)
        dot_y = Dot(point_y, color=WHITE, radius=0.06)

        label_coord = MathTex("x", color=WHITE, font_size=20).next_to(axes.c2p(x_val, 0), DOWN, buff=0.1)

        self.play(Create(line_x), FadeIn(dot_x))
        self.play(Create(line_y), FadeIn(dot_y))
        self.play(FadeIn(label_coord))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This calculates the probability density at any point z.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Highlight product by shading overlap area in YELLOW
        # We'll use a new curve representing the product
        def pdf_prod_func(x): return pdf_x_func(x) * pdf_y_shifted_func(x)
        curve_prod = axes.plot(pdf_prod_func, color=YELLOW, stroke_width=4)
        area_prod = axes.get_area(curve_prod, color=YELLOW, opacity=0.4)
        
        self.play(Create(curve_prod))
        self.play(FadeIn(area_prod))
        self.wait(2)
