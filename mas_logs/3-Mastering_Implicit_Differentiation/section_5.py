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
        title_str = "Visual Verification & Summary"
        lines_str = [
            "See the tangent line touch the curve precisely.",
            "The formula dy/dx equals negative x over y holds.",
            "Implicit differentiation finds slopes for complex shapes easily.",
            "Remember to always include the dy/dx tail.",
            "Now you can master any tangled relation."
        ]
        self.setup_layout(title_str, lines_str)

        # Colors for highlights
        color_1 = YELLOW_A
        color_2 = BLUE_A
        color_3 = GREEN_A
        color_4 = PINK
        color_5 = ORANGE

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_1))
        
        # Grid area for circle and axes
        # Positioning axes in the middle-right area (B3 to E5)
        axes = Axes(
            x_range=[-6, 6, 2],
            y_range=[-6, 6, 2],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": False, "font_size": 18, "color": GREY}
        )
        self.place_in_area(axes, 'B3', 'E5')
        
        # Radius 5 in axes units converted to scene units
        r_val = 5
        r_pixel = axes.coords_to_point(r_val, 0, 0)[0] - axes.coords_to_point(0, 0, 0)[0]
        circle = Circle(radius=r_pixel, color=BLUE_B)
        circle.move_to(axes.get_center())
        
        # Point and Tangent Line
        # Start angle for (3, 4) point
        theta = ValueTracker(np.arctan2(4, 3)) 
        
        def get_point_pos():
            return axes.coords_to_point(r_val * np.cos(theta.get_value()), r_val * np.sin(theta.get_value()))
        
        dot = always_redraw(lambda: Dot(get_point_pos(), color=YELLOW, radius=0.08))
        
        def get_tangent_line():
            p = get_point_pos()
            ang = theta.get_value()
            # Tangent direction vector is perpendicular to the radial vector
            direction = np.array([-np.sin(ang), np.cos(ang), 0])
            line = Line(p - direction * 1.5, p + direction * 1.5, color=GREEN_B, stroke_width=4)
            return line

        tangent_line = always_redraw(get_tangent_line)
        
        self.play(Create(axes), Create(circle))
        self.play(FadeIn(dot), Create(tangent_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color_2))
        
        # Formula display at the top of the interaction area
        formula_base = Text("dy/dx = -x/y", font_size=34)
        self.place_at_grid(formula_base, 'A2') # Positioned to leave space for calculation to the right
        
        def get_calc_text():
            x_val = r_val * np.cos(theta.get_value())
            y_val = r_val * np.sin(theta.get_value())
            # Avoid division by zero
            safe_y = y_val if abs(y_val) > 0.1 else (0.1 if y_val >= 0 else -0.1)
            slope = -x_val / safe_y
            calc = Text(f" ≈ -{x_val:.1f}/{y_val:.1f} = {slope:.2f}", font_size=28, color=BLUE_A)
            calc.next_to(formula_base, RIGHT, buff=0.2)
            return calc

        calculation = always_redraw(get_calc_text)
        
        self.play(Write(formula_base))
        self.play(FadeIn(calculation))
        
        # Animate the point moving to show calculation update
        self.play(theta.animate.set_value(theta.get_value() + PI/2), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color_3))
        # Continuous motion for emphasis
        self.play(theta.animate.set_value(theta.get_value() + PI/2), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(color_4))
        # Highlight the dy/dx term in the base formula
        highlight_box = SurroundingRectangle(formula_base, color=PINK, buff=0.1)
        self.play(Create(highlight_box))
        self.wait(2)
        self.play(FadeOut(highlight_box))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(color_5))
        
        # Summary steps group
        summary_title = Text("Key Steps:", font_size=26, color=WHITE).underline()
        summary_1 = Text("1. Treat y as a function of x", font_size=22, color=YELLOW_B)
        summary_2 = Text("2. Apply the Chain Rule", font_size=22, color=YELLOW_B)
        summary_3 = Text("3. Isolate the dy/dx term", font_size=22, color=YELLOW_B)
        summary_group = VGroup(summary_title, summary_1, summary_2, summary_3).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        
        # Place in the main visual area
        self.place_in_area(summary_group, 'C2', 'E5')
        
        # Clear specific visuals and show summary
        self.play(
            FadeOut(axes), FadeOut(circle), FadeOut(dot), 
            FadeOut(tangent_line), FadeOut(formula_base), FadeOut(calculation)
        )
        self.play(Write(summary_group))
        self.wait(3)
