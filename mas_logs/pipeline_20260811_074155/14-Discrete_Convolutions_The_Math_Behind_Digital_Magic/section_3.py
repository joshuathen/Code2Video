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
        # Data and titles
        title_str = "The Core Mechanism: Flip, Slide, Multiply, Sum"
        lecture_lines_str = [
            "First, we flip the kernel to account for time-reversal.",
            "Then, we slide the kernel across the input signal.",
            "At each step, we multiply overlapping values together.",
            "Next, we sum these products into one single value.",
            "This sum becomes a point in the output array."
        ]
        
        self.setup_layout(title_str, lecture_lines_str)
        
        # Colors based on storyboard
        SIGNAL_COLOR = "#FF8800"
        KERNEL_COLOR = "#00FF88"
        FLIP_COLOR = "#FFFF00"
        SLIDE_COLOR = "#FFFFFF"
        MULT_COLOR = "#FF0000"
        SUM_COLOR = "#0000FF"

        # Initialize data structures
        signal_vals = [1, 2, 3, 4, 5]
        kernel_vals = [1, 0, -1]
        
        def create_array(vals, color):
            group = VGroup()
            for v in vals:
                sq = Square(side_length=0.7, color=color)
                txt = Text(str(v), font_size=20, color=color)
                group.add(VGroup(sq, txt))
            return group.arrange(RIGHT, buff=0.1)

        signal_mobj = create_array(signal_vals, SIGNAL_COLOR)
        kernel_mobj = create_array(kernel_vals, KERNEL_COLOR)
        output_mobj = create_array(["?", "?", "?", "?", "?"], SUM_COLOR)

        # Initial placement (Fix layout based on Issue 35)
        self.place_in_area(signal_mobj, "B1", "B6")
        self.place_in_area(kernel_mobj, "C1", "C6", scale_factor=0.8)
        self.place_in_area(output_mobj, "E1", "E6", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # First, we flip the kernel to account for time-reversal.
        self.play(self.lecture[0].animate.set_color(FLIP_COLOR))
        self.play(FadeIn(signal_mobj), FadeIn(kernel_mobj))
        
        # Flip Animation (Rotation + Value Reverse)
        self.play(kernel_mobj.animate.set_color(FLIP_COLOR))
        
        flipped_vals = kernel_vals[::-1]
        flipped_mobj = create_array(flipped_vals, FLIP_COLOR)
        self.place_in_area(flipped_mobj, "C1", "C6", scale_factor=0.8)
        
        self.play(Rotate(kernel_mobj, angle=PI, axis=UP), run_time=1)
        self.play(Transform(kernel_mobj, flipped_mobj))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Then, we slide the kernel across the input signal.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(SLIDE_COLOR)
        )
        
        # Move kernel to align kernel[0] with signal[0]
        # Since they are both in row C/B area, they share the same horizontal center.
        # We want to shift kernel_mobj so its first element aligns horizontally with signal_mobj's first element.
        target_x = signal_mobj[0].get_x()
        current_x = kernel_mobj[0].get_x()
        shift_vec = np.array([target_x - current_x, 0, 0])
        
        self.play(
            kernel_mobj.animate.set_color(SLIDE_COLOR).shift(shift_vec),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # At each step, we multiply overlapping values together.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(MULT_COLOR)
        )
        
        # Show overlapping pairs flashing
        flash_rects = VGroup()
        for i in range(3):
            flash_rects.add(SurroundingRectangle(signal_mobj[i], color=MULT_COLOR, buff=0.05))
            flash_rects.add(SurroundingRectangle(kernel_mobj[i], color=MULT_COLOR, buff=0.05))
        
        calc_text = Text("1*(-1) + 2*0 + 3*1", font_size=24, color=MULT_COLOR)
        self.place_in_area(calc_text, "D1", "D6", scale_factor=0.7)
        
        self.play(Create(flash_rects))
        self.play(Write(calc_text))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Next, we sum these products into one single value.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(SUM_COLOR)
        )
        
        sum_result = Text(" = 2", font_size=24, color=SUM_COLOR)
        sum_result.next_to(calc_text, RIGHT)
        
        self.play(Write(sum_result))
        self.play(FadeOut(flash_rects))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This sum becomes a point in the output array.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(SUM_COLOR)
        )
        
        # Reveal output array and place value
        self.play(FadeIn(output_mobj))
        
        # Create the result value mobject
        res_val = Text("2", font_size=20, color=SUM_COLOR)
        # Position it inside a square at the output position
        final_val_box = VGroup(Square(side_length=0.7, color=SUM_COLOR), res_val)
        final_val_box.move_to(output_mobj[0])
        
        # Group the calculation for transformation
        calc_group = VGroup(calc_text, sum_result)
        
        self.play(
            Transform(calc_group, final_val_box),
            output_mobj[0].animate.set_opacity(0)
        )
        self.add(final_val_box)
        self.wait(2)
