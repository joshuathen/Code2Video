from manim import *
import numpy as np
import math

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
        # Setup the layout with title and lecture lines
        title = "Prerequisite Bridge: The Taylor Series"
        lines = [
            "Recall the scalar exponential function's Taylor Series expansion.",
            "Each polynomial term adds a layer of precision.",
            "This infinite sum defines the curve's perfect shape."
        ]
        self.setup_layout(title, lines)

        # Colors for the approximations
        colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create Coordinate System - Adjusted position as per Issue 30
        axes = Axes(
            x_range=[-2, 2.5, 1],
            y_range=[-1, 8, 1],
            axis_config={"include_tip": True, "font_size": 24},
            x_length=5,
            y_length=4
        )
        self.place_in_area(axes, "B3", "F6")
        
        # The target function e^x
        exp_curve = axes.plot(lambda x: np.exp(x), x_range=[-2, 2], color=WHITE, stroke_opacity=0.4)
        # Using Text to maintain consistency with previous local fixes
        exp_label = Text("e^x", color=WHITE, font_size=24)
        # Adjusted position and scale as per Issue 31
        self.place_at_grid(exp_label, "B6", scale_factor=0.8)

        # The Taylor Series Formula
        formula = Text(
            "e^x = 1 + x + x^2/2! + x^3/3! + ...",
            color=WHITE, font_size=24
        )
        # Adjusted area and scale as per Issue 32
        self.place_in_area(formula, "A2", "A6", scale_factor=0.7)

        self.play(Write(axes), run_time=1)
        self.play(Create(exp_curve), Write(exp_label), run_time=1)
        self.play(Write(formula), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Polynomial approximations
        def taylor_exp(x, n):
            return sum([x**i / math.factorial(i) for i in range(n + 1)])

        # Tracking variables for layers
        layers = VGroup()
        
        # Animate first 4 terms (n=0 to n=3)
        for n in range(4):
            curve = axes.plot(
                lambda x: taylor_exp(x, n),
                x_range=[-2, 2],
                color=colors[n]
            )
            # Label for the polynomial
            label = Text(f"P{n}(x)", color=colors[n], font_size=18)
            # Position label near the end of the curve relative to axes
            label.move_to(axes.c2p(1.8, taylor_exp(1.8, n) + 0.3))
            
            self.play(Create(curve), Write(label), run_time=1.5)
            layers.add(curve, label)
            self.wait(0.5)

        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Show higher order terms converging
        for n in range(4, 7):
            curve = axes.plot(
                lambda x: taylor_exp(x, n),
                x_range=[-2, 2.1],
                color=colors[n % len(colors)]
            )
            self.play(Create(curve), run_time=1)
            layers.add(curve)

        # Emphasize the final perfect curve
        self.play(
            exp_curve.animate.set_stroke(opacity=1.0, width=4),
            layers.animate.set_stroke(opacity=0.3),
            run_time=2
        )
        
        self.wait(2)
        self.lecture[2].set_color(WHITE)
        self.wait(1)
