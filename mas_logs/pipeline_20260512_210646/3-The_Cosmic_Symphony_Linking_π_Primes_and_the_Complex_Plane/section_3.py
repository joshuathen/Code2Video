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
            'Summing the squares of all reciprocal integers reveals a pattern.',
            'As terms stack, the total converges to a limit.',
            'This infinite sum unexpectedly reaches a specific value.',
            'Leonhard Euler proved it equals pi squared over six.',
            'Circles and integers are now mathematically tied together.'
        ]
        self.setup_layout("The Basel Problem: The First Bridge", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Summing the squares of all reciprocal integers reveals a pattern.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        sum_str = "1/1\u00b2 + 1/2\u00b2 + 1/3\u00b2 + 1/4\u00b2 + ..."
        sum_expr = Text(sum_str, font_size=36, color="#FFFFFF")
        # Fix for Issue 32
        self.place_in_area(sum_expr, "A1", "A4", scale_factor=0.7)
        self.play(Write(sum_expr))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # As terms stack, the total converges to a limit.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(TEAL)
        )
        
        # Draw horizontal segments stacked end-to-end
        segment_scale = 2.4
        start_x = self.grid["C1"][0]
        y_val = self.grid["C1"][1]
        
        # Basel limit is pi^2 / 6 approx 1.64493
        limit_x = start_x + (np.pi**2 / 6) * segment_scale
        limit_line = DashedLine(
            start=[limit_x, y_val + 1.0, 0],
            end=[limit_x, y_val - 1.0, 0],
            color="#00FF00"
        )
        limit_label = Text("Limit", font_size=16, color="#00FF00")
        self.place_at_grid(limit_label, "B5", scale_factor=1.0)
        limit_label.next_to(limit_line, UP, buff=0.1)
        
        self.play(Create(limit_line), FadeIn(limit_label))
        
        segments = VGroup()
        current_x = start_x
        # Create and animate 6 segments to show convergence
        for n in range(1, 7):
            length = (1 / (n**2)) * segment_scale
            seg = Line(
                start=[current_x, y_val, 0],
                end=[current_x + length, y_val, 0],
                stroke_width=10,
                color=interpolate_color(BLUE_D, TEAL, n/6)
            )
            segments.add(seg)
            current_x += length
            self.play(Create(seg), run_time=0.25)
        
        dots_visual = Text("...", font_size=24, color=WHITE)
        dots_visual.next_to(segments, RIGHT, buff=0.1)
        self.play(FadeIn(dots_visual))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This infinite sum unexpectedly reaches a specific value.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(ORANGE)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Leonhard Euler proved it equals pi squared over six.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#FFD700")
        )
        
        # Replace the sum expression with pi^2 / 6 in gold
        # We build result_expr as a group to target the pi symbol later
        pi_symbol = Text("\u03c0", color="#FFD700")
        rest_of_formula = Text("\u00b2 / 6", color="#FFD700")
        result_expr = VGroup(pi_symbol, rest_of_formula).arrange(RIGHT, buff=0.05)
        
        # Fix for Issues 33 & 34: avoid distortion and overlap
        self.place_at_grid(result_expr, "B2", scale_factor=0.7)
        
        self.play(ReplacementTransform(sum_expr, result_expr))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Circles and integers are now mathematically tied together.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(PINK)
        )
        
        # Surround the pi symbol with a small pulsating circle to emphasize geometric link
        pulse_circle = Circle(radius=0.4, color=WHITE, stroke_width=2)
        pulse_circle.move_to(pi_symbol.get_center())
        
        self.play(Create(pulse_circle))
        self.play(pulse_circle.animate.scale(1.3), rate_func=there_and_back, run_time=0.6)
        self.play(pulse_circle.animate.scale(1.3), rate_func=there_and_back, run_time=0.6)
        self.play(FadeOut(pulse_circle))
        self.wait(2)
