from manim import *
import numpy as np

# Fix for LaTeX cleanup race condition error
config.no_latex_cleanup = True

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
        # Section 2: The Core Concept: Local Scaling Factor
        title_text = "The Core Concept: Local Scaling Factor"
        lecture_lines = [
            "- Derivatives measure the local scaling factor of space.",
            "- Imagine a tiny segment dx on the input line.",
            "- This segment transforms into a new length dy.",
            "- The derivative is the ratio of these local lengths.",
            "- It tells us how much space stretches or squashes."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Define colors from storyboard
        color_dx = "#FF00FF"      # Magenta
        color_dy = "#FFFF00"      # Yellow
        color_formula = "#00FFFF" # Cyan
        color_text = "#FFFFFF"    # White

        # Initialize number lines
        input_line = NumberLine(x_range=[0, 5, 1], length=4.5, include_numbers=True, color=BLUE_B)
        output_line = NumberLine(x_range=[0, 15, 3], length=4.5, include_numbers=True, color=GREEN_B)
        
        input_label = Text("Input", font_size=24, color=BLUE_B)
        output_label = Text("Output", font_size=24, color=GREEN_B)

        # Positioning (using grid system)
        # Issue 29 Fix: shift input_line to B2-B6 to avoid overlap with label at B1
        self.place_in_area(input_line, "B2", "B6")
        self.place_at_grid(input_label, "B1", scale_factor=0.8)
        
        # Issue 30 Fix: shift output_line to E2-E6 to avoid overlap with label at E1
        self.place_in_area(output_line, "E2", "E6")
        self.place_at_grid(output_label, "E1", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # - Derivatives measure the local scaling factor of space.
        self.play(self.lecture[0].animate.set_color(color_text))
        scaling_text = Text("Local Scaling Factor", font_size=32, color=color_text)
        self.place_in_area(scaling_text, "A1", "A6")
        self.play(Write(scaling_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # - Imagine a tiny segment dx on the input line.
        self.play(self.lecture[1].animate.set_color(color_dx))
        
        dx_val = 0.5
        dx_start_num = 1.0
        dx_segment = Line(
            input_line.number_to_point(dx_start_num),
            input_line.number_to_point(dx_start_num + dx_val),
            color=color_dx, stroke_width=10
        )
        dx_label = MathTex("dx", color=color_dx, font_size=30)
        dx_label.next_to(dx_segment, UP, buff=0.1)
        
        self.play(Create(input_line), FadeIn(input_label))
        self.play(Create(dx_segment), FadeIn(dx_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # - This segment transforms into a new length dy.
        self.play(self.lecture[2].animate.set_color(color_dy))
        
        dy_start_num = 3.0
        dy_length_initial = dx_val 
        dy_length_final = 1.5
        
        dy_segment = Line(
            output_line.number_to_point(dy_start_num),
            output_line.number_to_point(dy_start_num + dy_length_initial),
            color=color_dy, stroke_width=10
        )
        dy_label = MathTex("dy", color=color_dy, font_size=30)
        dy_label.next_to(dy_segment, DOWN, buff=0.1)
        
        self.play(Create(output_line), FadeIn(output_label))
        self.play(Create(dy_segment), FadeIn(dy_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # - The derivative is the ratio of these local lengths.
        self.play(self.lecture[3].animate.set_color(color_formula))
        
        stretch_tracker = ValueTracker(dy_length_initial)
        
        dy_segment.add_updater(
            lambda m: m.set_points_as_corners([
                output_line.number_to_point(dy_start_num),
                output_line.number_to_point(dy_start_num + stretch_tracker.get_value())
            ])
        )
        dy_label.add_updater(lambda m: m.next_to(dy_segment, DOWN, buff=0.1))
        
        formula = MathTex(
            r"f'(x) = \frac{dy}{dx} = \frac{1.5}{0.5} = 3", 
            color=color_formula, font_size=36
        )
        # Issue 31 Fix: Move to C1-D6 to utilize Row C and avoid cramping
        self.place_in_area(formula, "C1", "D6")
        
        self.play(
            stretch_tracker.animate.set_value(dy_length_final),
            Write(formula),
            run_time=2
        )
        self.wait(1)
        
        dy_segment.clear_updaters()
        dy_label.clear_updaters()

        # === Animation for Lecture Line 5 ===
        # - It tells us how much space stretches or squashes.
        self.play(self.lecture[4].animate.set_color(color_text))
        
        highlight_box = SurroundingRectangle(formula, color=color_formula, buff=0.1)
        self.play(Create(highlight_box))
        self.play(Indicate(dy_segment, color=color_dy, scale_factor=1.2))
        self.play(FadeOut(highlight_box))
        
        self.wait(3)
