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
        self.setup_layout(
            "The Bridge: The Fundamental Theorem of Calculus",
            [
                "Differentiation and integration are actually inverse processes.",
                "This link is the Fundamental Theorem of Calculus.",
                "Integrating then differentiating returns the original function.",
                "It connects slopes and areas through a single bridge.",
                "Calculus is a two-way street between these concepts."
            ]
        )

        # Colors
        BLUE_COLOR = "#0000FF"
        GREEN_COLOR = "#00FF00"
        WHITE_COLOR = "#FFFFFF"
        YELLOW_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE_COLOR)
        integral_expr = MathTex(r"\int_{a}^{x} f(t) \, dt", color=BLUE_COLOR)
        self.place_in_area(integral_expr, "B2", "C5", scale_factor=1.2)
        
        self.play(Write(integral_expr))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(GREEN_COLOR)
        
        # Derivative operator around the integral
        deriv_op_full = MathTex(r"\frac{d}{dx}", r"\left[", r"\int_{a}^{x} f(t) \, dt", r"\right]")
        deriv_op_full[0].set_color(GREEN_COLOR)
        deriv_op_full[1].set_color(GREEN_COLOR)
        deriv_op_full[2].set_color(BLUE_COLOR)
        deriv_op_full[3].set_color(GREEN_COLOR)
        
        self.place_in_area(deriv_op_full, "B2", "C5", scale_factor=1.2)
        self.play(Transform(integral_expr, deriv_op_full))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE_COLOR)
        
        # Final equation: d/dx [...] = f(x)
        # Fix for Issue 38: use 'B2' to 'C5'
        final_eq = MathTex(
            r"\frac{d}{dx}", r"\left[", r"\int_{a}^{x} f(t) \, dt", r"\right]", r"=", r"f(x)"
        )
        final_eq[0].set_color(GREEN_COLOR)
        final_eq[1].set_color(GREEN_COLOR)
        final_eq[2].set_color(BLUE_COLOR)
        final_eq[3].set_color(GREEN_COLOR)
        final_eq[4].set_color(WHITE_COLOR)
        final_eq[5].set_color(WHITE_COLOR)
        
        self.place_in_area(final_eq, "B2", "C5", scale_factor=1.0)
        
        self.play(Transform(integral_expr, final_eq))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(YELLOW_COLOR)
        
        # Issue 32 & Issue 40: SVGMobject and positioning 'E3'-'F4'
        factory_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/factory.svg")
        factory_svg.set_color(GREY)
        factory_label = Text("FTC Factory", font_size=16, color=WHITE)
        factory_group = VGroup(factory_svg, factory_label).arrange(DOWN, buff=0.1)
        self.place_in_area(factory_group, "E3", "F4", scale_factor=1.0)
        
        # Area representation (Input) - Issue 39: use 'E2'
        area_rect = Rectangle(width=0.8, height=0.5, color=BLUE_COLOR, fill_opacity=0.8)
        area_label = Text("Area", font_size=14, color=BLUE_COLOR)
        area_in = VGroup(area_rect, area_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(area_in, "E2")
        
        # Height representation (Output) - Issue 40: use 'E5'
        height_line = Line(ORIGIN, UP * 0.8, color=WHITE_COLOR, stroke_width=4)
        height_label = Text("f(x)", font_size=14, color=WHITE_COLOR)
        height_out = VGroup(height_line, height_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(height_out, "E5")

        self.play(Create(factory_group))
        self.play(FadeIn(area_in))
        
        # Move Area into factory, then output f(x)
        self.play(area_in.animate.move_to(factory_group.get_center()))
        self.play(FadeOut(area_in))
        self.play(FadeIn(height_out, shift=RIGHT))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE_COLOR)
        
        # Target the f(x) part of the equation for the glow
        target = final_eq[5]
        glows = VGroup(*[
            SurroundingRectangle(target, color=WHITE, buff=0.1 + i*0.05, stroke_width=2)
            for i in range(3)
        ])
        
        # Use rate_functions.there_and_back (Belief B058)
        self.play(Create(glows))
        self.play(
            glows.animate.set_stroke(width=10, opacity=0.3),
            run_time=1,
            rate_func=rate_functions.there_and_back
        )
        self.play(FadeOut(glows))
        self.wait(2)
