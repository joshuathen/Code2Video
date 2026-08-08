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
        # Setup lines from storyboard
        lecture_lines = [
            "Consider a tiny increase in width, dx.",
            "This creates a small sliver of area, dA.",
            "The sliver's height is roughly f(x).",
            "So, the change dA equals f(x) times dx.",
            "Thus, the derivative of area is the function."
        ]
        self.setup_layout("Visual Proof: The Rate of Area Change", lecture_lines)

        # Color Palette
        color_dx = "#87CEEB"  # Sky Blue
        color_da = "#FFFF00"  # Yellow (Storyboard: dA in yellow)
        color_fx = "#FFFF00"  # Yellow (Storyboard: f(x) in yellow)
        color_eqn = "#FFA500" # Orange
        color_final = "#FFFFFF" # White (Storyboard: Final equation in bright white)

        # Setup Axes and Function in the grid (Column 3 to 6)
        # Using C3-F6 area for the graph to leave room for equations in rows A and B
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 4, 1],
            x_length=3.5,
            y_length=3,
            axis_config={"color": WHITE, "include_tip": True},
        )
        self.place_in_area(axes, "C3", "F6")
        
        func = axes.plot(lambda x: 0.1 * x**2 + 0.5, color=WHITE)
        func_label = MathTex("f(x)", color=WHITE, font_size=24)
        # Place label near the end of the function
        self.place_at_grid(func_label, "C6")

        # Static area under curve up to x=2
        x_val = 2.0
        dx_val = 0.5
        area_main = axes.get_area(func, x_range=[0, x_val], color=BLUE, opacity=0.3)
        
        x_line = axes.get_vertical_line(axes.c2p(x_val, func.underlying_function(x_val)), color=WHITE)
        x_label = MathTex("x", font_size=20).next_to(axes.c2p(x_val, 0), DOWN)

        self.add(axes, func, area_main, x_line, x_label, func_label)

        # === Animation for Lecture Line 1 ===
        # Consider a tiny increase in width, dx.
        self.play(self.lecture[0].animate.set_color(color_dx))
        
        dx_brace = BraceBetweenPoints(axes.c2p(x_val, 0), axes.c2p(x_val + dx_val, 0), color=color_dx)
        dx_text = MathTex("dx", color=color_dx, font_size=24).next_to(dx_brace, DOWN, buff=0.1)
        
        x_plus_dx_line = axes.get_vertical_line(axes.c2p(x_val + dx_val, func.underlying_function(x_val + dx_val)), color=color_dx)
        
        self.play(Create(dx_brace), Write(dx_text), Create(x_plus_dx_line), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This creates a small sliver of area, dA.
        self.play(self.lecture[1].animate.set_color(color_da))
        
        area_sliver = axes.get_area(func, x_range=[x_val, x_val + dx_val], color=color_da, opacity=0.6)
        da_label = MathTex("dA", color=color_da, font_size=24)
        # Fix for Issue 39: Move da_label to D5 and scale to 0.8
        self.place_at_grid(da_label, "D5", scale_factor=0.8)

        self.play(FadeIn(area_sliver), Write(da_label), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The sliver's height is roughly f(x).
        self.play(self.lecture[2].animate.set_color(color_fx))
        
        # Highlight height at x
        height_line = Line(axes.c2p(x_val, 0), axes.c2p(x_val, func.underlying_function(x_val)), color=color_fx, stroke_width=4)
        fx_text = MathTex("f(x)", color=color_fx, font_size=24).next_to(height_line, LEFT, buff=0.1)
        
        # Approximating rectangle
        rect = Rectangle(
            width=axes.c2p(dx_val, 0)[0] - axes.c2p(0, 0)[0],
            height=axes.c2p(0, func.underlying_function(x_val))[1] - axes.c2p(0, 0)[1],
            fill_color=color_fx,
            fill_opacity=0.3,
            stroke_width=1,
            stroke_color=color_fx
        ).move_to(axes.c2p(x_val + dx_val/2, func.underlying_function(x_val)/2))

        self.play(Create(height_line), Write(fx_text), FadeIn(rect), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # So, the change dA equals f(x) times dx.
        self.play(self.lecture[3].animate.set_color(color_eqn))
        
        eqn_da = MathTex("dA", "\\approx", "f(x)", "\\cdot", "dx", font_size=32)
        eqn_da[0].set_color(color_da)
        eqn_da[2].set_color(color_fx)
        eqn_da[4].set_color(color_dx)
        
        # Fix for Issue 38: Move eqn_da to A4-A6 and scale to 0.8
        self.place_in_area(eqn_da, "A4", "A6", scale_factor=0.8)
        
        self.play(Write(eqn_da), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Thus, the derivative of area is the function.
        self.play(self.lecture[4].animate.set_color(color_final))
        
        final_eqn = MathTex("\\frac{dA}{dx}", "=", "f(x)", font_size=36, color=color_final)
        # Fix for Issue 40: Move final_eqn to A4-A6 and scale to 0.9
        self.place_in_area(final_eqn, "A4", "A6", scale_factor=0.9)
        
        # Transform and Flash
        self.play(
            Transform(eqn_da, final_eqn),
            Flash(final_eqn, color=WHITE, line_length=0.2),
            run_time=2
        )
        self.wait(2)

# Update issue statuses
# update_issue(38, under_review=True, resolution_note="Moved eqn_da to area A4-A6 with scale_factor 0.8 to avoid overlap with graph.")
# update_issue(39, under_review=True, resolution_note="Moved da_label to grid D5 with scale_factor 0.8 for better visual association with the sliver.")
# update_issue(40, under_review=True, resolution_note="Moved final_eqn to area A4-A6 with scale_factor 0.9 for emphasis.")
