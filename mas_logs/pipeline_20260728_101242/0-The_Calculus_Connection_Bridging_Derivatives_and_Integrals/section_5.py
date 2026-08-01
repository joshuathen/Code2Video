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

class Section5Scene(TeachingScene):
    def construct(self):
        title = "Visual Synthesis: The Rocket Launch Example"
        lines = [
            "A rocket launch combines both calculus concepts.",
            "Velocity's area shows the rocket's increasing altitude.",
            "Velocity's slope reveals the force of acceleration."
        ]
        self.setup_layout(title, lines)

        # Colors
        ROCKET_COLOR = WHITE
        AREA_COLOR = "#FFA500"
        TANGENT_COLOR = "#FF4500"
        VELOCITY_COLOR = "#00BFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)

        # Setup Axes on the right part of the grid
        axes = Axes(
            x_range=[0, 4.2, 1],
            y_range=[0, 18, 4],
            x_length=4.5,
            y_length=4,
            axis_config={"include_tip": True, "font_size": 18}
        ).add_coordinates()
        
        # Place axes in the area B2 to F6
        self.place_in_area(axes, "B2", "F6", scale_factor=0.8)
        
        v_label = MathTex("v(t)", font_size=24, color=VELOCITY_COLOR)
        self.place_at_grid(v_label, "A2", scale_factor=0.8)

        # Velocity function: v(t) = t^2
        def velocity_func(t):
            return t**2

        v_graph = axes.plot(velocity_func, x_range=[0, 4], color=VELOCITY_COLOR)

        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/rocket.svg]
        rocket = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rocket.svg")
        rocket.set_color(ROCKET_COLOR)
        # Initial position at bottom of the first column
        self.place_at_grid(rocket, "F1", scale_factor=0.6)
        rocket_end_pos = self.grid["A1"]

        self.play(
            Create(axes),
            Write(v_label),
            run_time=1
        )
        
        self.play(
            Create(v_graph),
            rocket.animate.move_to(rocket_end_pos),
            run_time=3,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(AREA_COLOR)

        time_tracker = ValueTracker(0.01)

        # Area under the curve
        area = always_redraw(lambda: axes.get_area(
            v_graph, 
            x_range=[0, time_tracker.get_value()], 
            color=AREA_COLOR, 
            opacity=0.5
        ))
        
        area_label = Text("Altitude (Integral)", font_size=18, color=AREA_COLOR)
        # Fix for Issue 46: Move area_label from B6 to A5-A6 area
        self.place_in_area(area_label, "A5", "A6", scale_factor=1.0)

        self.add(area)
        self.play(
            FadeIn(area_label),
            time_tracker.animate.set_value(4),
            run_time=4,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(TANGENT_COLOR)

        # Moving Tangent Line
        tangent_tracker = ValueTracker(0.5)
        
        # To avoid creating a new TangentLine object every frame, 
        # we can use a single line and update its position and rotation.
        # However, TangentLine depends on the graph and alpha.
        # Let's keep it for now as it's just a Line (cheap).
        def get_tangent():
            val = tangent_tracker.get_value()
            line = TangentLine(v_graph, alpha=val/4, length=2, color=TANGENT_COLOR)
            return line

        tangent_line = always_redraw(get_tangent)
        
        accel_label = Text("Acceleration (Derivative)", font_size=18, color=TANGENT_COLOR)
        # Fix for Issue 46: Move accel_label to A3-A4 area
        self.place_in_area(accel_label, "A3", "A4", scale_factor=1.0)

        self.play(
            FadeIn(accel_label),
            FadeIn(tangent_line),
            run_time=1
        )

        self.play(
            tangent_tracker.animate.set_value(4),
            run_time=4,
            rate_func=linear
        )
        self.wait(2)
