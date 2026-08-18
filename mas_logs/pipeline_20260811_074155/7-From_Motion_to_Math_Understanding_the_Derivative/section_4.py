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
        self.setup_layout("The Limit: The Magic of Zero", [
            "If h equals zero, the calculation fails.",
            "We cannot divide by zero in mathematics.",
            "Instead, we use a limit to approach zero.",
            "The secant line transforms into a tangent line.",
            "This tangent line touches the curve at one point."
        ])

        # === Animation for Lecture Line 1 ===
        # Display the expression (f(x+h) - f(x)) / h in white (#FFFFFF).
        self.lecture[0].set_color(YELLOW)
        
        # Using a single string to ensure stable rendering
        formula = MathTex(r"\frac{f(x+h) - f(x)}{h}", color="#FFFFFF")
        # Issue 31: Reduce scale factor to 1.0 for better spacing
        self.place_in_area(formula, 'B2', 'C5', scale_factor=1.0)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Pulse the denominator h in red (#FF0000) to highlight the division by zero problem.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Accessing the symbol mobject for the denominator 'h'
        # In the expression \frac{f(x+h) - f(x)}{h}, 'h' is the last glyph.
        h_denom = formula[0][-1]
        
        self.play(
            h_denom.animate.set_color("#FF0000").scale(1.5),
            run_time=0.4
        )
        self.play(h_denom.animate.scale(1/1.5), run_time=0.4)
        self.play(h_denom.animate.scale(1.5), run_time=0.4)
        self.play(h_denom.animate.scale(1/1.5), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show the limit notation lim (h->0) appearing in front of the expression.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        limit_notation = MathTex(r"\lim_{h \to 0}", color="#FFFFFF")
        # Position relative to current formula
        limit_notation.next_to(formula, LEFT, buff=0.2)
        
        full_limit_group = VGroup(limit_notation, formula)
        
        # Center the combined expression in the upper area
        self.play(
            FadeIn(limit_notation),
            full_limit_group.animate.move_to(self.grid["B3"] + RIGHT * 0.7)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Animate the blue secant line rotating until it becomes a red tangent line (#FF0000) at x.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 7, 2],
            x_length=4,
            y_length=3,
            axis_config={"include_tip": False, "color": GRAY}
        )
        # Issue 32: Reduce scale factor to 0.8 to avoid crowding
        self.place_in_area(axes, 'D2', 'F5', scale_factor=0.8)
        
        # Parabola for demonstration
        curve = axes.plot(lambda x: x**2, x_range=[0, 2.5], color=WHITE)
        
        x_val = 1.0
        h_tracker = ValueTracker(1.2)
        
        # Secant line initially blue (#00BFFF)
        secant_line = Line(color="#00BFFF")
        
        def update_secant(mob):
            h = h_tracker.get_value()
            if abs(h) < 0.001: h = 0.001  # Stability for slope
            p1 = axes.c2p(x_val, x_val**2)
            p2 = axes.c2p(x_val + h, (x_val + h)**2)
            v = p2 - p1
            # Extend the line visually across the axes area
            mob.set_points_as_corners([p1 - v*2.5, p1 + v*4.5])

        secant_line.add_updater(update_secant)
        
        self.play(Create(axes), Create(curve))
        self.play(Create(secant_line))
        self.wait(1)
        
        # Animate h approaching zero to simulate the limit process
        self.play(h_tracker.animate.set_value(0.001), run_time=3)
        self.wait(0.5)
        
        # Define the exact tangent line at x=1 (Slope = 2, Equation: y = 2x - 1)
        p_tan_start = axes.c2p(0.4, 2*0.4 - 1)
        p_tan_end = axes.c2p(2.2, 2*2.2 - 1)
        tangent_line = Line(p_tan_start, p_tan_end, color="#FF0000").set_stroke(width=5)
        
        # Turn off updater before transformation
        secant_line.remove_updater(update_secant)
        self.play(
            Transform(secant_line, tangent_line),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Add a label "Tangent Line: Instantaneous Slope" (#FF0000) near the point of tangency.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        tangent_label = Text("Tangent Line: Instantaneous Slope", font_size=18, color="#FF0000")
        # Issue 30: Move to F5 and reduce scale to 0.6 to avoid overlap
        self.place_at_grid(tangent_label, 'F5', scale_factor=0.6)
        
        self.play(Write(tangent_label))
        self.wait(2)
