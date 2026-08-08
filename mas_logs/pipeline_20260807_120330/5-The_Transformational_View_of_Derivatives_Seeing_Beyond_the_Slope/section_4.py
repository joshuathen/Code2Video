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
        # Initial Setup
        title_text = "The Direction of Transformation"
        lecture_lines = [
            "- The sign of the derivative determines the mapping's direction.",
            "- A positive derivative preserves the original orientation of space.",
            "- A negative derivative flips the space in the output."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_INPUT = "#FFFFFF"
        COLOR_POSITIVE = "#00FF00"
        COLOR_NEGATIVE = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # "The sign of the derivative determines the mapping's direction."
        self.lecture[0].set_color(COLOR_INPUT)

        # Input and Output Lines
        # Using columns 3-6 for the lines to maintain gap from lecture text (B021)
        input_line = NumberLine(x_range=[-3, 3, 1], length=3.5, include_numbers=True, font_size=18, color=COLOR_INPUT)
        output_line = NumberLine(x_range=[-6, 6, 2], length=3.5, include_numbers=True, font_size=18, color=COLOR_INPUT)
        
        self.place_in_area(input_line, 'B3', 'B6')
        self.place_in_area(output_line, 'E3', 'E6')
        
        input_label = Text("Input", font_size=20, color=COLOR_INPUT)
        output_label = Text("Output", font_size=20, color=COLOR_INPUT)
        
        # Position labels offset from the movement area (B013)
        self.place_at_grid(input_label, 'B2', scale_factor=0.8)
        self.place_at_grid(output_label, 'E2', scale_factor=0.8)

        # Initial function display
        func_label = MathTex("f(x) = 2x", font_size=36, color=COLOR_POSITIVE)
        # Fix Issue 35: Move to center of rows C and D
        self.place_in_area(func_label, 'C3', 'D6')

        self.play(
            Create(input_line),
            Create(output_line),
            Write(input_label),
            Write(output_label),
            Write(func_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "A positive derivative preserves the original orientation of space."
        self.lecture[1].set_color(COLOR_POSITIVE)

        # ValueTracker for smooth movement
        t = ValueTracker(0)

        # Create arrows that will be updated by the ValueTracker
        input_arrow = Arrow(color=COLOR_INPUT, stroke_width=4, buff=0)
        output_arrow = Arrow(color=COLOR_POSITIVE, stroke_width=4, buff=0)

        def update_input_arrow(mob):
            val = t.get_value()
            mob.put_start_and_end_on(input_line.n2p(val), input_line.n2p(val + 0.5))

        def update_output_arrow_pos(mob):
            val = t.get_value()
            # f(x) = 2x mapping
            mob.put_start_and_end_on(output_line.n2p(2 * val), output_line.n2p(2 * (val + 0.5)))

        input_arrow.add_updater(update_input_arrow)
        output_arrow.add_updater(update_output_arrow_pos)

        self.add(input_arrow, output_arrow)
        # Animate from 0 to 1 on the input line
        self.play(t.animate.set_value(1), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "A negative derivative flips the space in the output."
        self.lecture[2].set_color(COLOR_NEGATIVE)

        # Transition to f(x) = -2x
        new_func_label = MathTex("f(x) = -2x", font_size=36, color=COLOR_NEGATIVE)
        # Fix Issue 36: Move to center of rows C and D
        self.place_in_area(new_func_label, 'C3', 'D6')

        # Reset ValueTracker and update the output arrow behavior
        self.play(
            FadeOut(input_arrow),
            FadeOut(output_arrow),
            Transform(func_label, new_func_label),
            run_time=0.5
        )
        
        t.set_value(0)
        output_arrow.remove_updater(update_output_arrow_pos)
        output_arrow.set_color(COLOR_NEGATIVE)

        def update_output_arrow_neg(mob):
            val = t.get_value()
            # f(x) = -2x mapping
            # Input forward: [val, val+0.5] -> Output: [-2*val, -2*(val+0.5)]
            # Since -2*(val+0.5) < -2*val, the arrow points left (reversal)
            mob.put_start_and_end_on(output_line.n2p(-2 * val), output_line.n2p(-2 * (val + 0.5)))

        # Highlight reversal of orientation with a brief flash (FF0000)
        flash = Flash(output_line.n2p(0), color=COLOR_NEGATIVE, flash_radius=0.5)
        
        self.play(flash)
        
        input_arrow.add_updater(update_input_arrow)
        output_arrow.add_updater(update_output_arrow_neg)
        
        self.play(
            FadeIn(input_arrow),
            FadeIn(output_arrow)
        )
        
        # Animate movement showing the flipped direction
        self.play(t.animate.set_value(1), run_time=2, rate_func=linear)
        self.wait(2)

        # Final cleanup
        input_arrow.remove_updater(update_input_arrow)
        output_arrow.remove_updater(update_output_arrow_neg)
