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
        # Define lecture lines
        lecture_lines = [
            "Gradient descent nudges weights to reduce loss.",
            "We move weights opposite to the gradient direction.",
            "The learning rate determines our step size.",
            "We descend toward the point of minimum error.",
            "The network optimizes itself one step at a time."
        ]
        
        # Setup layout
        self.setup_layout("Gradient Descent: Nudging the Knobs", lecture_lines)

        # Colors for highlighting and elements
        highlight_color = YELLOW
        ball_color = "#00FF00"
        curve_color = "#4477AA"
        gradient_color = "#FF4444"
        step_color = "#FFEE00"
        knob_color = "#888888"

        # Assets
        ball_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg"
        knob_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/knob.svg"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(highlight_color)
        
        # Cost Landscape: Parabola-like curve
        def landscape_func(x):
            return 0.5 * (x - 3.0)**2 - 0.5
        
        curve = FunctionGraph(landscape_func, x_range=[0.5, 5.5], color=curve_color)
        
        # Ball [Asset: ball.svg] sits on the curve
        x_start = 1.0
        ball = SVGMobject(ball_asset_path).scale(0.2)
        ball.set_color(ball_color)
        ball.move_to(np.array([x_start, landscape_func(x_start), 0]))
        
        self.play(Create(curve), FadeIn(ball))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(highlight_color)
        
        # Calculate slope for tangent/gradient: f'(x) = (x - 3.0)
        slope = (x_start - 3.0)
        
        grad_arrow = Arrow(
            start=ball.get_center(),
            end=ball.get_center() + 1.2 * np.array([1, slope, 0]),
            color=gradient_color,
            buff=0,
            stroke_width=4
        )
        grad_label = Text("Gradient", font_size=16, color=gradient_color).next_to(grad_arrow, RIGHT, buff=0.1)
        
        step_arrow = Arrow(
            start=ball.get_center(),
            end=ball.get_center() - 0.8 * np.array([1, slope, 0]),
            color=step_color,
            buff=0,
            stroke_width=6
        )
        step_label = Text("Nudge", font_size=16, color=step_color).next_to(step_arrow, LEFT, buff=0.1)

        self.play(GrowArrow(grad_arrow), FadeIn(grad_label))
        self.wait(0.5)
        self.play(GrowArrow(step_arrow), FadeIn(step_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(highlight_color)
        
        # Fix Issue 34: Center lr_text from E1 to E6
        lr_text = Text("Step Size = Learning Rate \u00d7 Gradient", font_size=18, color=step_color)
        self.place_in_area(lr_text, "E1", "E6", scale_factor=0.8)
        
        self.play(Write(lr_text))
        self.play(step_arrow.animate.scale(1.2), run_time=0.5, rate_func=there_and_back)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(highlight_color)
        
        # Movement step
        x_step1 = 2.0
        target_pos1 = np.array([x_step1, landscape_func(x_step1), 0])
        
        self.play(
            FadeOut(grad_arrow, grad_label, step_arrow, step_label, lr_text),
            ball.animate.move_to(target_pos1),
            run_time=2,
            rate_func=smooth
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(highlight_color)
        
        # Fix Issue 25 & 35: Integrated Weight knob [Asset: knob.svg] and repositioned to F5-F6
        knob = SVGMobject(knob_asset_path).set_color(knob_color)
        self.place_in_area(knob, "F5", "F6", scale_factor=1.0)
        knob_label = Text("Weight Knob", font_size=16).next_to(knob, UP, buff=0.2)
        
        self.play(FadeIn(knob, knob_label))
        
        # Final descent step + knob turn
        x_final = 3.0
        final_pos = np.array([x_final, landscape_func(x_final), 0])
        
        self.play(
            ball.animate.move_to(final_pos),
            Rotate(knob, angle=-PI/2),
            run_time=2
        )
        
        self.wait(2)
