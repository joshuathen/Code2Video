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
        self.setup_layout("Derivative as a Scaling Factor", [
            "The derivative is a local scaling factor.",
            "Moving dx in input moves f'(x)*dx in output.",
            "It measures how much space stretches or squishes.",
            "f'(x) = 3 means the segment stretches threefold.",
            "Mapping transforms the local density of the input."
        ])

        # === Animation for Lecture Line 1 ===
        # Display Input and Output number lines
        self.lecture[0].set_color(YELLOW)
        
        input_line = NumberLine(x_range=[-2, 2, 1], length=4, include_tip=True, color=GRAY)
        output_line = NumberLine(x_range=[-2, 2, 1], length=4, include_tip=True, color=GRAY)
        
        input_label = Text("Input", font_size=18, color=GRAY)
        output_label = Text("Output", font_size=18, color=GRAY)
        
        self.place_in_area(input_line, 'B1', 'B6')
        self.place_in_area(output_line, 'E1', 'E6')
        
        input_label.next_to(input_line, LEFT, buff=0.2)
        output_label.next_to(output_line, LEFT, buff=0.2)
        
        # Small interval 'dx' on Input line
        dx_val = 0.4
        x_tracker = ValueTracker(-1.0)
        
        # Start points don't matter much because updater takes over
        dx_interval = Line(
            input_line.number_to_point(-0.2), 
            input_line.number_to_point(0.2), 
            color="#00FFFF", stroke_width=8
        )
        dx_interval.add_updater(lambda m: m.move_to(input_line.number_to_point(x_tracker.get_value())))
        
        dx_label = MathTex("dx", color="#00FFFF", font_size=24)
        dx_label.add_updater(lambda m: m.next_to(dx_interval, UP, buff=0.1))
        
        self.play(Create(input_line), Create(output_line), Write(input_label), Write(output_label))
        self.play(Create(dx_interval), Write(dx_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show the mapped interval on the Output line
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Derivative value tracker (f'(x))
        deriv_tracker = ValueTracker(3.0)
        
        df_interval = Line(
            output_line.number_to_point(-0.6), 
            output_line.number_to_point(0.6), 
            color="#FF00FF", stroke_width=8
        )
        
        def update_df(m):
            center_x = x_tracker.get_value()
            half_len = (dx_val * deriv_tracker.get_value()) / 2
            # Use set_points_as_corners to redraw the segment with new length
            m.set_points_as_corners([
                output_line.number_to_point(center_x - half_len),
                output_line.number_to_point(center_x + half_len)
            ])

        df_interval.add_updater(update_df)
        
        df_label = MathTex("df = f'(x)dx", color="#FF00FF", font_size=24)
        df_label.add_updater(lambda m: m.next_to(df_interval, DOWN, buff=0.1))
        
        self.play(Create(df_interval), Write(df_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display the equation 'f'(x) = df / dx'
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        eqn = MathTex("f'(x) = \\frac{df}{dx}", color="#FFFF00", font_size=32)
        # Fix for Issue #27: Moving equation to D5 to avoid obstruction
        self.place_at_grid(eqn, 'D5', scale_factor=0.9)
        
        # Connecting lines (visual mapping)
        mapping_line_l = Line(color=GRAY, stroke_width=1, stroke_opacity=0.5)
        mapping_line_r = Line(color=GRAY, stroke_width=1, stroke_opacity=0.5)
        
        mapping_line_l.add_updater(lambda m: m.set_points_as_corners([dx_interval.get_left(), df_interval.get_left()]))
        mapping_line_r.add_updater(lambda m: m.set_points_as_corners([dx_interval.get_right(), df_interval.get_right()]))
        
        self.play(Write(eqn), Create(mapping_line_l), Create(mapping_line_r))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # f'(x) = 3 means the segment stretches threefold.
        # Animate dx moving and f'(x) changing.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Move dx and vary derivative
        self.play(
            x_tracker.animate.set_value(1.0),
            deriv_tracker.animate.set_value(0.5),
            run_time=2,
            rate_func=linear
        )
        self.play(
            x_tracker.animate.set_value(-0.5),
            deriv_tracker.animate.set_value(4.0),
            run_time=2,
            rate_func=linear
        )
        
        # === Animation for Lecture Line 5 ===
        # Flash 'Scaling Factor' when stretching is most prominent
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        scaling_text = Text("Scaling Factor", color="#FF8800", font_size=36)
        # Fix for Issue #28: Moving scaling_text to A5 for better composition
        self.place_at_grid(scaling_text, 'A5', scale_factor=0.8)
        
        self.play(
            deriv_tracker.animate.set_value(5.0),
            Flash(scaling_text, color="#FF8800", flash_radius=1.5),
            Write(scaling_text),
            run_time=1
        )
        self.play(FadeOut(scaling_text))
        
        self.play(
            x_tracker.animate.set_value(0),
            deriv_tracker.animate.set_value(3.0),
            run_time=2
        )
        self.wait(2)
