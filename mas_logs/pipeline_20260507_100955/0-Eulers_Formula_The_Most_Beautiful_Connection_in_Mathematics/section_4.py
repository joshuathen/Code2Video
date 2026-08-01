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
        # Issue 44: Updated lecture lines to match final script (max 10 words each)
        lecture_lines = [
            'Imaginary growth traces a perfect unit circle.',
            'Horizontal distance is captured by the cosine.',
            'Vertical height is tracked by the sine.',
            'As the angle grows, the point rotates.',
            'This gives us the famous Euler’s formula.'
        ]
        self.setup_layout("The Visual Derivation: Walking the Unit Circle", lecture_lines)
        
        # Define colors
        COLOR_CIRCLE = "#FFD700" # Gold
        COLOR_COS = "#00FF00"    # Green
        COLOR_SIN = "#FF0000"    # Red
        COLOR_TEXT = "#FFFFFF"   # White
        
        # Angle tracker (radians)
        angle_tracker = ValueTracker(0.001)
        
        # Create circle - Issue 32, 33, 44: Reposition to C2-F6, scale 0.8
        circle = Circle(radius=2, color=COLOR_CIRCLE)
        self.place_in_area(circle, "C2", "F6", scale_factor=0.8)
        center = circle.get_center()

        # Helper for current point on circle
        def get_curr_point():
            return circle.point_at_angle(angle_tracker.get_value())

        # Persistent mobjects for updaters
        dot = Dot(color=COLOR_TEXT)
        dot.add_updater(lambda d: d.move_to(get_curr_point()))

        # Label for the point e^ix - Issue 32 fix (offset ensures no overlap with top formula)
        point_label = Text("e^ix", color=COLOR_TEXT, font_size=24)
        point_label.add_updater(lambda m: m.next_to(dot, UR, buff=0.15))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(circle))
        self.add(dot, point_label)
        self.play(angle_tracker.animate.set_value(PI/6), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Green horizontal line (cosine)
        cos_line = Line(color=COLOR_COS, stroke_width=5)
        cos_line.add_updater(lambda l: l.set_points_as_corners([
            center, 
            [get_curr_point()[0], center[1], 0]
        ]))
        
        cos_text = Text("cos(x)", color=COLOR_COS, font_size=20)
        cos_text.add_updater(lambda m: m.move_to([
            (center[0] + get_curr_point()[0]) / 2,
            center[1] - 0.3,
            0
        ]))

        self.play(Create(cos_line), Write(cos_text))
        self.play(angle_tracker.animate.set_value(PI/4), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Red vertical line (sine)
        sin_line = Line(color=COLOR_SIN, stroke_width=5)
        sin_line.add_updater(lambda l: l.set_points_as_corners([
            [get_curr_point()[0], center[1], 0],
            get_curr_point()
        ]))

        sin_text = Text("i sin(x)", color=COLOR_SIN, font_size=20)
        sin_text.add_updater(lambda m: m.move_to([
            get_curr_point()[0] + 0.6,
            (center[1] + get_curr_point()[1]) / 2,
            0
        ]))

        self.play(Create(sin_line), Write(sin_text))
        self.play(angle_tracker.animate.set_value(PI/3), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Angle arc and label
        angle_arc = Arc(radius=0.5, start_angle=0, angle=angle_tracker.get_value(), arc_center=center, color=WHITE)
        angle_arc.add_updater(lambda a: a.become(Arc(
            radius=0.5, 
            start_angle=0, 
            angle=angle_tracker.get_value(), 
            arc_center=center, 
            color=WHITE
        )))
        
        angle_label = Text("x", color=WHITE, font_size=24)
        angle_label.add_updater(lambda m: m.move_to(center + 0.75 * np.array([
            np.cos(angle_tracker.get_value()/2), 
            np.sin(angle_tracker.get_value()/2), 
            0
        ])))

        self.play(Create(angle_arc), Write(angle_label))
        # Point rotates as angle grows
        self.play(angle_tracker.animate.set_value(3*PI/4), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Display main formula - Issue 34, 44: Reposition to area A2-A6, scale 0.8
        formula = Text("e^ix = cos(x) + i sin(x)", color=COLOR_TEXT, font_size=28)
        self.place_in_area(formula, 'A2', 'A6', scale_factor=0.8)
        
        self.play(Write(formula))
        # Final rotation to emphasize the circular path and the connection
        self.play(angle_tracker.animate.set_value(2*PI + PI/6), run_time=6, rate_func=linear)
        self.wait(2)
