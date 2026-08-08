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

class Section6Scene(TeachingScene):
    def construct(self):
        # Define content
        title = "The Update: Gradient Descent"
        lecture_lines = [
            "Now, we take a small step against the gradient.",
            "We subtract a fraction of the gradient from weights.",
            "This fraction is controlled by the \"Learning Rate.\"",
            "Small steps prevent overcorrecting and ensure stable learning.",
            "Repeating this process thousands of times achieves high accuracy."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Color palette
        COLOR_GRADIENT = "#00FFFF"  # Cyan
        COLOR_WEIGHT = "#7CFC00"    # LawnGreen
        COLOR_ERROR = "#FFA500"     # Orange

        # === Animation for Lecture Line 1 ===
        # Now, we take a small step against the gradient.
        self.lecture[0].set_color(COLOR_GRADIENT)
        
        # Weight display: W = 10.0
        w_label = MathTex("W =", color=WHITE)
        w_value = DecimalNumber(10.0, num_decimal_places=1, color=WHITE)
        weight_group = VGroup(w_label, w_value).arrange(RIGHT, buff=0.1)
        self.place_at_grid(weight_group, "C4", scale_factor=0.8)
        
        # Machine knob: Circle with a pointer
        # Fix 38: knob at C4 to enclose weight
        knob_circle = Circle(radius=0.7, color=WHITE)
        knob_line = Line(knob_circle.get_center(), knob_circle.get_top(), color=WHITE)
        knob = VGroup(knob_circle, knob_line)
        self.place_at_grid(knob, "C4", scale_factor=1.0)
        
        self.play(FadeIn(knob), FadeIn(weight_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We subtract a fraction of the gradient from weights.
        self.lecture[1].set_color(COLOR_GRADIENT)
        
        # Visualize subtraction
        # Fix 40: minus_sign at B4, scale 0.6
        minus_sign = MathTex("-", color=COLOR_GRADIENT)
        self.place_at_grid(minus_sign, "B4", scale_factor=0.6)
        
        # Learning rate factor appearing
        lr_factor = MathTex("0.02 \\times \\nabla", color=COLOR_GRADIENT).scale(0.7)
        self.place_at_grid(lr_factor, "B5")
        
        self.play(Write(minus_sign), FadeIn(lr_factor))
        self.play(Indicate(weight_group, color=COLOR_GRADIENT))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This fraction is controlled by the "Learning Rate."
        self.lecture[2].set_color(COLOR_WEIGHT)
        
        # Turning the knob and decreasing weight
        # 10.0 -> 9.8, change color to green
        self.play(
            Rotate(knob, angle=-PI/4, about_point=knob_circle.get_center()),
            w_value.animate.set_value(9.8).set_color(COLOR_WEIGHT),
            FadeOut(minus_sign),
            FadeOut(lr_factor),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Small steps prevent overcorrecting and ensure stable learning.
        self.lecture[3].set_color(COLOR_ERROR)
        
        # Mini-hill and arrow (visualizing small step down slope)
        hill = Arc(radius=2, start_angle=PI/4, angle=PI/2, color=WHITE)
        self.place_at_grid(hill, "E4", scale_factor=0.5)
        
        # Arrow pointing down the slope
        arrow = Arrow(start=hill.get_start(), end=hill.get_center(), color=COLOR_GRADIENT, buff=0.1)
        
        self.play(Create(hill), GrowArrow(arrow))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Repeating this process thousands of times achieves high accuracy.
        self.lecture[4].set_color(COLOR_ERROR)
        
        # Cleanup hill/arrow
        self.play(FadeOut(hill), FadeOut(arrow))
        
        # Error Gap bracket: vertical bracket moved to column 6 to avoid overlap
        # Fix 39: area 'B6' to 'E6'
        top_p = self.grid["B6"]
        bot_p = self.grid["E6"]
        bracket_span = Line(top_p, bot_p, stroke_opacity=0)
        error_brace = Brace(bracket_span, RIGHT, color=COLOR_ERROR)
        error_text = Text("Error Gap", font_size=20, color=COLOR_ERROR)
        error_text.next_to(error_brace, RIGHT, buff=0.1)
        error_viz = VGroup(error_brace, error_text)
        
        self.play(Create(error_brace), FadeIn(error_text))
        self.wait(1)
        
        # Shrink the error gap visual significantly to show convergence
        target_center = self.grid["C6"]
        self.play(
            error_viz.animate.scale(0.1).move_to(target_center),
            run_time=2
        )
        
        # Final confirmation
        success_check = MathTex(r"\checkmark", color=COLOR_WEIGHT).scale(2)
        self.place_at_grid(success_check, "C5")
        self.play(Write(success_check))
        self.wait(3)
