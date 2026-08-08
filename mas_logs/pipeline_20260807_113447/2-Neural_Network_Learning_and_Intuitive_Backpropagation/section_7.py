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

class Section7Scene(TeachingScene):
    def construct(self):
        # Colors for lecture lines and matching elements
        COLOR_1 = YELLOW
        COLOR_2 = BLUE
        COLOR_3 = "#00FF00" # Pure green as requested

        self.setup_layout("Summary: The Iterative Loop", [
            "- Repeat: Guess, Check, Blame, and Update.",
            "- Thousands of iterations refine the internal knobs.",
            "- Eventually, the machine achieves perfect prediction."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(COLOR_1))

        # Show a circular arrow loop icon labeled 'The Training Loop'
        # Create a loop using two CurvedArrows
        arrow1 = CurvedArrow(UP*0.5, DOWN*0.5, angle=PI*0.8, color=COLOR_1).shift(LEFT*0.1)
        arrow2 = CurvedArrow(DOWN*0.5, UP*0.5, angle=PI*0.8, color=COLOR_1).shift(RIGHT*0.1)
        loop_icon = VGroup(arrow1, arrow2)
        self.place_in_area(loop_icon, 'C3', 'C4', scale_factor=0.7)
        
        loop_label = Text("The Training Loop", font_size=20, color=COLOR_1)
        self.place_in_area(loop_label, 'B3', 'B4', scale_factor=0.8)

        self.play(Create(loop_icon), Write(loop_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_2)
        )

        # The 'Error Meter' setup
        meter_bg = Rectangle(width=2, height=0.4, color=WHITE)
        meter_fill = Rectangle(width=1.8, height=0.3, color=RED, fill_opacity=0.8, stroke_width=0)
        meter_fill.align_to(meter_bg, LEFT).shift(RIGHT*0.1)
        meter_group = VGroup(meter_bg, meter_fill)
        self.place_in_area(meter_group, 'E3', 'E5', scale_factor=1.0)
        
        meter_label = Text("Error Meter", font_size=20, color=COLOR_2)
        self.place_at_grid(meter_label, 'E2', scale_factor=0.8)

        # The 'Neural Network' knobs
        def create_knob(color=BLUE):
            circle = Circle(radius=0.3, color=color)
            center = circle.get_center()
            line = Line(center, center + UP * 0.3, color=color)
            return VGroup(circle, line)

        knobs = VGroup(*[create_knob() for _ in range(5)])
        knob_positions = ['D2', 'D3', 'D4', 'D5', 'D6']
        for knob, pos in zip(knobs, knob_positions):
            self.place_at_grid(knob, pos, scale_factor=0.7)

        self.play(FadeIn(meter_group), FadeIn(meter_label), FadeIn(knobs))
        
        # Updaters for the "training process"
        error_tracker = ValueTracker(1.0)
        # Update width of meter_fill based on tracker
        meter_fill.add_updater(lambda m: m.stretch_to_fit_width(max(0.01, error_tracker.get_value() * 1.8), about_edge=LEFT))
        
        # Jitter updater for lines (the "knob pointers")
        def jitter_line(line):
            line.rotate(np.random.uniform(-0.4, 0.4), about_point=line.get_start())

        for k in knobs:
            k[1].add_updater(jitter_line)

        # Rotate the loop icon to show activity
        loop_icon.add_updater(lambda m, dt: m.rotate(dt * 3))

        # Perform the "thousands of iterations" animation
        self.play(
            error_tracker.animate.set_value(0.2),
            run_time=4,
            rate_func=linear
        )
        
        # Stop jittering as we approach the end of iterations
        for k in knobs:
            k[1].remove_updater(jitter_line)
        
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_3)
        )

        # Eventually, the machine achieves perfect prediction.
        # Error meter turns green and hits zero. Knobs settle.
        target_rotations = [PI/4, -PI/3, PI/2, -PI/6, PI]
        
        self.play(
            error_tracker.animate.set_value(0.01),
            meter_fill.animate.set_color(COLOR_3),
            *[Rotate(knobs[i][1], angle=target_rotations[i], about_point=knobs[i][1].get_start()) for i in range(5)],
            run_time=2
        )
        
        # Stop loop rotation
        loop_icon.remove_updater(loop_icon.updaters[0])
        
        self.wait(3)
