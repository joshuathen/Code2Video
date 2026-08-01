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
        # Title and Lecture Lines
        title_text = "Trigonometric Derivatives: The Rhythm of the Circle"
        lecture_lines = [
            "A rider travels around this circular Ferris wheel.",
            "Their vertical position follows a smooth sine wave.",
            "Their vertical speed matches the circle's horizontal cosine."
        ]
        self.setup_layout(title_text, lecture_lines)

        # --- Coordinate System & Objects ---
        # State tracker for the rotation angle
        theta_tracker = ValueTracker(0)
        radius_val = 1.2
        
        # Load and place Ferris Wheel Asset
        # [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/ferriswheel.svg]
        ferris_wheel = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/ferriswheel.svg")
        ferris_wheel.set_color(WHITE)
        self.place_in_area(ferris_wheel, "B3", "E5", scale_factor=1.2)
        circle_center = ferris_wheel.get_center()

        # Visual aid circle and axes
        circle = Circle(radius=radius_val, color=WHITE, stroke_opacity=0.3)
        circle.move_to(circle_center)
        
        axes = VGroup(
            Line(circle_center + LEFT * 1.5, circle_center + RIGHT * 1.5, color=GRAY_B, stroke_opacity=0.4),
            Line(circle_center + DOWN * 1.5, circle_center + UP * 1.5, color=GRAY_B, stroke_opacity=0.4)
        )
        
        # Persistent mobjects with updaters
        radius_line = Line(circle_center, circle_center, color=WHITE)
        def update_radius(m):
            t = theta_tracker.get_value()
            m.put_start_and_end_on(
                circle_center,
                circle_center + np.array([radius_val * np.cos(t), radius_val * np.sin(t), 0])
            )
        radius_line.add_updater(update_radius)

        rider = Dot(color=WHITE, radius=0.08)
        def update_rider(m):
            t = theta_tracker.get_value()
            m.move_to(circle_center + np.array([radius_val * np.cos(t), radius_val * np.sin(t), 0]))
        rider.add_updater(update_rider)

        # Ferris wheel rotation updater
        last_theta = [0]
        def update_wheel(m):
            curr_theta = theta_tracker.get_value()
            m.rotate(curr_theta - last_theta[0])
            last_theta[0] = curr_theta
        ferris_wheel.add_updater(update_wheel)

        # Green sine line (vertical height)
        sine_line = Line(color="#00FF00", stroke_width=6)
        def update_sine(m):
            t = theta_tracker.get_value()
            x = radius_val * np.cos(t)
            y = radius_val * np.sin(t)
            m.put_start_and_end_on(
                circle_center + np.array([x, 0, 0]),
                circle_center + np.array([x, y, 0])
            )
        sine_line.add_updater(update_sine)

        # Blue cosine vector (horizontal component representing vertical velocity magnitude)
        cosine_vector = Arrow(color="#0000FF", buff=0, stroke_width=4, max_tip_length_to_length_ratio=0.2)
        def update_cosine(m):
            t = theta_tracker.get_value()
            x = radius_val * np.cos(t)
            m.put_start_and_end_on(
                circle_center,
                circle_center + np.array([x, 0, 0])
            )
        cosine_vector.add_updater(update_cosine)

        # Labels and Formulas
        # Issue 36: Move sin_label to A5
        sin_label = Text("sin(θ)", color="#00FF00", font_size=24)
        self.place_at_grid(sin_label, "A5", scale_factor=0.8)
        
        # Issue 37: Move cos_label to E6
        cos_label = Text("cos(θ)", color="#0000FF", font_size=24)
        self.place_at_grid(cos_label, "E6", scale_factor=0.8)
        
        # Issue 38: Move deriv_formula to area F3-F6
        deriv_formula = Text("d/dθ sin(θ) = cos(θ)", font_size=32, color=WHITE)
        self.place_in_area(deriv_formula, "F3", "F6", scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        # "A rider travels around this circular Ferris wheel."
        self.lecture[0].set_color(WHITE)
        self.add(axes, ferris_wheel, circle, radius_line, rider)
        # One full loop to establish the movement
        self.play(theta_tracker.animate.set_value(2 * PI), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Their vertical position follows a smooth sine wave."
        self.lecture[1].set_color("#00FF00")
        self.add(sine_line)
        self.play(Write(sin_label))
        # Highlight the vertical movement
        self.play(theta_tracker.animate.set_value(4 * PI), run_time=4, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Their vertical speed matches the circle's horizontal cosine."
        self.lecture[2].set_color("#0000FF")
        self.add(cosine_vector)
        self.play(Write(cos_label))
        self.play(Write(deriv_formula))
        # Final loops to observe the relationship
        self.play(theta_tracker.animate.set_value(6 * PI), run_time=4, rate_func=linear)
        self.wait(2)
