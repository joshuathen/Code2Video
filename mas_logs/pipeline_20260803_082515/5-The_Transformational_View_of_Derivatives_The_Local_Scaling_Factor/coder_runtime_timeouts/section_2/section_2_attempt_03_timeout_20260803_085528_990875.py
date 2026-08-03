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

class Section2Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Linear functions like y equals 3x scale space uniformly.",
            "Every interval is stretched by a constant factor.",
            "The slope is simply this constant stretching factor."
        ]
        self.setup_layout("Prerequisite: Linear Transformations", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # Formula y = 3x
        formula = MathTex("y = 3x", color=YELLOW)
        self.place_in_area(formula, "A2", "A5", scale_factor=1.2)
        
        # Number Lines
        input_line = NumberLine(x_range=[-2, 2, 1], length=5, include_numbers=True, color=BLUE_B)
        output_line = NumberLine(x_range=[-6, 6, 3], length=5, include_numbers=True, color=GREEN_B)
        
        input_label = Text("Input (x)", font_size=18, color=BLUE_B)
        output_label = Text("Output (y)", font_size=18, color=GREEN_B)
        
        # Position using grid
        self.place_in_area(input_line, "B1", "B6", scale_factor=1.0)
        self.place_in_area(output_line, "D1", "D6", scale_factor=1.0)
        
        input_label.next_to(input_line, LEFT, buff=0.2)
        output_label.next_to(output_line, LEFT, buff=0.2)

        self.play(Write(formula))
        self.play(Create(input_line), FadeIn(input_label))
        self.play(Create(output_line), FadeIn(output_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )

        # Small interval dx
        dx_val = 0.4
        x_start = -1.5
        x_tracker = ValueTracker(x_start)
        
        dx_rect = Line(input_line.n2p(x_start), input_line.n2p(x_start + dx_val), color=BLUE, stroke_width=8)
        dy_rect = Line(output_line.n2p(3 * x_start), output_line.n2p(3 * (x_start + dx_val)), color=GREEN, stroke_width=8)
        
        dx_label = MathTex("dx", color=BLUE, font_size=24)
        dy_label = MathTex("dy", color=GREEN, font_size=24)

        arrow_start = Arrow(input_line.n2p(x_start), output_line.n2p(3 * x_start), buff=0.1, color=GRAY, stroke_width=2)
        arrow_end = Arrow(input_line.n2p(x_start + dx_val), output_line.n2p(3 * (x_start + dx_val)), buff=0.1, color=GRAY, stroke_width=2)

        def update_dx_rect(mob):
            x = x_tracker.get_value()
            mob.set_points_as_corners([input_line.n2p(x), input_line.n2p(x + dx_val)])
            
        def update_dy_rect(mob):
            x = x_tracker.get_value()
            mob.set_points_as_corners([output_line.n2p(3 * x), output_line.n2p(3 * (x + dx_val))])

        def update_arrow_start(mob):
            x = x_tracker.get_value()
            mob.put_start_and_end_on(input_line.n2p(x), output_line.n2p(3 * x))
            
        def update_arrow_end(mob):
            x = x_tracker.get_value()
            mob.put_start_and_end_on(input_line.n2p(x + dx_val), output_line.n2p(3 * (x + dx_val)))

        dx_rect.add_updater(update_dx_rect)
        dy_rect.add_updater(update_dy_rect)
        arrow_start.add_updater(update_arrow_start)
        arrow_end.add_updater(update_arrow_end)
        
        dx_label.add_updater(lambda m: m.next_to(dx_rect, UP, buff=0.1))
        dy_label.add_updater(lambda m: m.next_to(dy_rect, DOWN, buff=0.1))

        self.play(Create(dx_rect), FadeIn(dx_label))
        self.play(GrowArrow(arrow_start), GrowArrow(arrow_end))
        self.play(Create(dy_rect), FadeIn(dy_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )

        # Formula dy = 3 * dx
        factor_text = MathTex("dy = 3 \\cdot dx", color=YELLOW)
        self.place_in_area(factor_text, "F2", "F5", scale_factor=1.0)
        
        self.play(Write(factor_text))
        
        # Move interval across the line
        self.play(x_tracker.animate.set_value(1.0), run_time=4, rate_func=linear)
        self.wait(1)
        
        self.play(x_tracker.animate.set_value(-1.5), run_time=2, rate_func=linear)
        self.wait(2)
