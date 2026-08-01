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
        # Initial Setup
        lecture_lines = [
            "Some curves, like the Folium, loop and cross themselves.",
            "Solving for y is nearly impossible for this knot.",
            "Apply the product rule to the term six x y.",
            "This formula gives the slope anywhere on the loop.",
            "The tangent line glides smoothly around the complex curve."
        ]
        self.setup_layout("Complex Curves: The Folium of Descartes", lecture_lines)

        # Define Colors
        FOLIUM_COLOR = "#FF00FF"
        PRODUCT_RULE_COLOR = "#FFFF00"
        DERIVATIVE_COLOR = "#FFFFFF"

        # Define Math elements using Text (to avoid LaTeX issues)
        folium_eq = Text("x^3 + y^3 = 6xy", font_size=32, color=FOLIUM_COLOR)
        product_eq = Text("3x^2 + 3y^2(dy/dx) = 6y + 6x(dy/dx)", font_size=24, color=PRODUCT_RULE_COLOR)
        final_eq = Text("dy/dx = (6y - 3x^2) / (3y^2 - 6x)", font_size=24, color=DERIVATIVE_COLOR)

        # Position Equation 1 (Issue 43 fix)
        self.place_in_area(folium_eq, 'A2', 'B5', scale_factor=0.8)

        # Define Coordinate System
        axes = Axes(
            x_range=[-1, 5, 1],
            y_range=[-1, 5, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": BLUE_D, "include_tip": True}
        )
        
        # Folium of Descartes Parameterization: x = 6t/(1+t^3), y = 6t^2/(1+t^3)
        # Loop is roughly t in [0, 10]
        folium_curve = ParametricFunction(
            lambda t: axes.c2p(6*t/(1+t**3), 6*t**2/(1+t**3)),
            t_range=[0, 20, 0.05],
            color=FOLIUM_COLOR
        )
        
        # Group axes and curve for grid placement (Issue 44 fix)
        axes_group = VGroup(axes, folium_curve)
        self.place_in_area(axes_group, 'C1', 'E6', scale_factor=0.9)

        # Position final derivative (Issue 45 fix)
        self.place_in_area(final_eq, 'F2', 'F5', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        # Some curves, like the Folium, loop and cross themselves.
        self.lecture[0].set_color(FOLIUM_COLOR)
        self.play(Write(folium_eq))
        self.play(Create(axes), Create(folium_curve), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Solving for y is nearly impossible for this knot.
        self.lecture[1].set_color(WHITE)
        # Indication of zoom/complexity (scale up effect on the loop part)
        # Since we use grid system, we can't easily use camera zoom without affecting layout.
        # We will briefly highlight the loop area.
        highlight_circle = Circle(radius=0.8, color=WHITE, stroke_width=2).move_to(axes.c2p(2, 2))
        self.play(Create(highlight_circle))
        self.play(FadeOut(highlight_circle))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Apply the product rule to the term six x y.
        self.lecture[2].set_color(PRODUCT_RULE_COLOR)
        # Position product rule equation relative to others
        self.place_at_grid(product_eq, 'B3', scale_factor=1.0)
        self.play(Write(product_eq))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This formula gives the slope anywhere on the loop.
        self.lecture[3].set_color(DERIVATIVE_COLOR)
        self.play(Write(final_eq))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The tangent line glides smoothly around the complex curve.
        self.lecture[4].set_color(WHITE)
        
        t_tracker = ValueTracker(0.2) # Start slightly after origin
        
        def get_tangent_line():
            t = t_tracker.get_value()
            # Coordinates
            px = 6*t / (1 + t**3)
            py = 6*t**2 / (1 + t**3)
            # Derivatives
            dx_dt = 6 * (1 - 2 * t**3) / (1 + t**3)**2
            dy_dt = 6 * t * (2 - t**3) / (1 + t**3)**2
            
            # Avoid division by zero
            if abs(dx_dt) < 0.001:
                slope = 1000 # vertical
            else:
                slope = dy_dt / dx_dt
            
            # Create a line of length 2 centered at (px, py)
            line_angle = np.arctan(slope)
            p_start = axes.c2p(px - np.cos(line_angle), py - np.sin(line_angle))
            p_end = axes.c2p(px + np.cos(line_angle), py + np.sin(line_angle))
            
            return Line(p_start, p_end, color=WHITE, stroke_width=4)

        tangent = always_redraw(get_tangent_line)
        dot = always_redraw(lambda: Dot(axes.c2p(
            6*t_tracker.get_value()/(1+t_tracker.get_value()**3),
            6*t_tracker.get_value()**2/(1+t_tracker.get_value()**3)
        ), color=WHITE))

        self.add(tangent, dot)
        # Gliding through the loop (t=0 to roughly t=10 covers the loop and returning to origin)
        self.play(t_tracker.animate.set_value(5.0), run_time=5, rate_func=linear)
        self.wait(2)
