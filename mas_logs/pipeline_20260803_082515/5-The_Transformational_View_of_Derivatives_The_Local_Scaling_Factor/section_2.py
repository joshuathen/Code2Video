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
        # Lecture lines from storyboard
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
        self.place_at_grid(formula, "A3", scale_factor=1.0)
        
        # Create input and output axes
        input_line = NumberLine(x_range=[-2, 2, 1], length=5, include_numbers=True, color=WHITE)
        input_label = Text("Input x", font_size=18, color=WHITE).next_to(input_line, LEFT, buff=0.3)
        input_group = VGroup(input_line, input_label)
        
        output_line = NumberLine(x_range=[-6, 6, 3], length=5, include_numbers=True, color=WHITE)
        output_label = Text("Output y", font_size=18, color=WHITE).next_to(output_line, LEFT, buff=0.3)
        output_group = VGroup(output_line, output_label)
        
        axes_group = VGroup(input_group, output_group).arrange(DOWN, buff=1.5)
        # Issue 23: Position axes_group in the designated area
        self.place_in_area(axes_group, 'A1', 'D6', scale_factor=0.6)
        
        self.play(Write(formula))
        self.play(Create(axes_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second lecture line
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(GREEN)
        )
        
        dx_val = 0.5
        x_tracker = ValueTracker(-1.5)
        
        # Intervals and connecting lines
        dx_line = Line(color=BLUE, stroke_width=8)
        dy_line = Line(color=GREEN, stroke_width=8)
        dx_label = MathTex("dx", color=BLUE, font_size=24)
        dy_label = MathTex("dy", color=GREEN, font_size=24)
        
        c1 = Line(color=GRAY, stroke_width=1, stroke_opacity=0.5)
        c2 = Line(color=GRAY, stroke_width=1, stroke_opacity=0.5)

        # Function to sync positions (not used as updater yet, just for initial setup)
        def sync_viz():
            x = x_tracker.get_value()
            p1 = input_line.n2p(x)
            p2 = input_line.n2p(x + dx_val)
            p3 = output_line.n2p(3 * x)
            p4 = output_line.n2p(3 * (x + dx_val))
            
            dx_line.set_points_as_corners([p1, p2])
            dy_line.set_points_as_corners([p3, p4])
            c1.set_points_as_corners([p1, p3])
            c2.set_points_as_corners([p2, p4])
            dx_label.next_to(dx_line, UP, buff=0.1)
            dy_label.next_to(dy_line, DOWN, buff=0.1)

        sync_viz()
        
        self.play(
            Create(dx_line),
            Create(dy_line),
            FadeIn(dx_label),
            FadeIn(dy_label),
            Create(c1),
            Create(c2)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third lecture line
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(BLUE)
        )
        
        # Issue 24: Create and place scaling factor box
        sc_formula = MathTex("dy = 3 \\cdot dx", color=BLUE)
        sc_rect = SurroundingRectangle(sc_formula, color=BLUE, buff=0.1)
        scaling_factor_box = VGroup(sc_formula, sc_rect)
        self.place_in_area(scaling_factor_box, 'E1', 'F3', scale_factor=0.8)
        
        # Issue 25: Create and place stretch text
        stretch_text = Text("Constant Stretching", font_size=22, color=WHITE)
        self.place_in_area(stretch_text, 'E4', 'F6', scale_factor=0.8)

        # Attach updaters for smooth interval movement
        dx_line.add_updater(lambda m: m.set_points_as_corners([input_line.n2p(x_tracker.get_value()), input_line.n2p(x_tracker.get_value() + dx_val)]))
        dy_line.add_updater(lambda m: m.set_points_as_corners([output_line.n2p(3 * x_tracker.get_value()), output_line.n2p(3 * (x_tracker.get_value() + dx_val))]))
        c1.add_updater(lambda m: m.set_points_as_corners([input_line.n2p(x_tracker.get_value()), output_line.n2p(3 * x_tracker.get_value())]))
        c2.add_updater(lambda m: m.set_points_as_corners([input_line.n2p(x_tracker.get_value() + dx_val), output_line.n2p(3 * (x_tracker.get_value() + dx_val))]))
        dx_label.add_updater(lambda m: m.next_to(dx_line, UP, buff=0.1))
        dy_label.add_updater(lambda m: m.next_to(dy_line, DOWN, buff=0.1))

        self.play(FadeIn(scaling_factor_box))
        # Move the dx interval across the input range
        self.play(x_tracker.animate.set_value(1.0), run_time=4, rate_func=linear)
        self.play(FadeIn(stretch_text))
        self.wait(2)
        
        # Cleanup updaters
        for mob in [dx_line, dy_line, c1, c2, dx_label, dy_label]:
            mob.clear_updaters()
