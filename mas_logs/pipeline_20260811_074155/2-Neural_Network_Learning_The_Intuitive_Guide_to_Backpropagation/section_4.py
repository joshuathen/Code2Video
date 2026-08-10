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
        # TEACHING CONTENT
        title_text = "The Loss Function: Measuring the 'Ouch'"
        lecture_lines = [
            "Loss measures the gap between prediction and reality.",
            "A large gap means a high error or \"ouch.\"",
            "We visualize this loss as a deep valley."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Initial state: all lecture lines are greyed out
        self.lecture.set_color(GRAY)

        # === Animation for Lecture Line 1 ===
        # Draw a white parabolic curve (Loss Curve) across the center.
        # Defined as y = 0.4x^2 to fit the area nicely
        loss_curve = FunctionGraph(
            lambda x: 0.4 * x**2,
            x_range=[-2.2, 2.2],
            color=WHITE
        )
        # Position in a central-right area
        self.place_in_area(loss_curve, "B2", "E6", scale_factor=0.9)
        
        self.play(
            Create(loss_curve),
            self.lecture[0].animate.set_color(WHITE),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show a vertical line between 'Guess' and 'Reality' labeled 'Loss' (#FF4500).
        
        # Bottom of the parabola (Reality)
        reality_pt = loss_curve.get_bottom()
        
        # A point on the high slope (Guess)
        # proportion 0.15 puts it high on the left branch
        guess_pt = loss_curve.point_from_proportion(0.15)
        
        # Create a vertical line for the loss gap
        # From the guess point down to the x-coordinate of the reality level
        loss_target = np.array([guess_pt[0], reality_pt[1], 0])
        loss_line = Line(guess_pt, loss_target, color="#FF4500", stroke_width=6)
        
        # Labels for the components
        # Relative positioning for labels within 1 grid unit (approx 1.0 units)
        loss_label = Text("Loss", font_size=20, color="#FF4500").next_to(loss_line, LEFT, buff=0.15)
        guess_label = Text("Guess", font_size=18, color=WHITE).next_to(guess_pt, UP, buff=0.2)
        reality_label = Text("Reality", font_size=18, color=WHITE).next_to(reality_pt, DOWN, buff=0.2)
        
        self.play(
            Create(loss_line),
            Write(loss_label),
            Write(guess_label),
            Write(reality_label),
            self.lecture[1].animate.set_color("#FF4500"),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Place a red ball (#FF0000) on the high slope of the curve.
        
        red_ball = Dot(radius=0.15, color="#FF0000")
        red_ball.move_to(guess_pt)
        
        self.play(
            FadeIn(red_ball),
            self.lecture[2].animate.set_color("#FF0000"),
            run_time=1.5
        )
        
        # Small roll animation to emphasize the valley shape
        self.play(
            red_ball.animate.move_to(loss_curve.point_from_proportion(0.2)),
            run_time=1,
            rate_func=bezier([0, 0, 1, 1])
        )
        
        self.wait(3)
