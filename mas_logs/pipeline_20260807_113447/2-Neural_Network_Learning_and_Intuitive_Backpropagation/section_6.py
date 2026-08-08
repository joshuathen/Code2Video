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
        # Fetch data from storyboard
        title = "Gradient Descent: The Small Step Forward"
        lines = [
            "We follow the error slope downward.",
            "Gradient descent updates weights in small steps.",
            "Learning rate prevents overshooting the solution.",
            "Tiny adjustments gradually improve Byte's accuracy.",
            "Each step brings us closer to the bottom."
        ]
        
        self.setup_layout(title, lines)

        # Colors
        SLOPE_COLOR = "#00FF7F" # Spring Green
        TANGENT_COLOR = "#FFD700" # Gold
        BALL_COLOR = "#FFFFFF" # White
        SLIDER_COLOR = "#00BFFF" # Deep Sky Blue

        # Assets
        BALL_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg"

        # === Animation for Lecture Line 1 ===
        # Line 1: "We follow the error slope downward."
        # Visual: Show a ball [Asset] on a curve (slope). Draw a tangent arrow (#FFD700) pointing down.
        self.lecture[0].set_color(SLOPE_COLOR)
        
        # Define the curve (loss landscape approximation)
        curve = FunctionGraph(
            lambda x: 0.3 * x**2,
            x_range=[-2.5, 2.5],
            color=SLOPE_COLOR
        )
        # Position curve in the lower right area
        self.place_in_area(curve, "C2", "F6", scale_factor=1.0)
        
        # Ball using Asset integration (Issue #26)
        ball_x = ValueTracker(2.0)
        
        def get_ball_pos():
            return curve.get_point_from_function(ball_x.get_value())

        ball = SVGMobject(BALL_ASSET).set_color(BALL_COLOR).scale(0.3)
        ball.move_to(get_ball_pos())
        # Use updater for efficient movement (Instruction 10)
        ball.add_updater(lambda m: m.move_to(get_ball_pos()))
        
        # Tangent arrow using persistent mobject and updater
        tangent_arrow = Arrow(buff=0, color=TANGENT_COLOR, stroke_width=4, max_tip_length_to_length_ratio=0.3)
        def update_arrow(arr):
            curr_x = ball_x.get_value()
            # f(x) = 0.3x^2 -> f'(x) = 0.6x
            slope = 0.6 * curr_x
            angle = np.arctan(slope)
            # Arrow points in the direction of negative gradient
            direction = np.array([-np.cos(angle), -np.sin(angle), 0])
            start = get_ball_pos()
            end = start + direction * 0.8
            arr.put_start_and_end_on(start, end)
        
        tangent_arrow.add_updater(update_arrow)

        self.play(Create(curve), run_time=1.5)
        self.play(FadeIn(ball), run_time=0.5)
        self.add(tangent_arrow)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: "Gradient descent updates weights in small steps."
        self.lecture[1].set_color(BALL_COLOR)
        # The ball takes a small step
        self.play(ball_x.animate.set_value(1.5), run_time=1.5, rate_func=smooth)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: "Learning rate prevents overshooting the solution."
        self.lecture[2].set_color(SLIDER_COLOR)
        
        # Fix placement of slider_label per Issue #36
        slider_label = Text("Learning Rate", font_size=18, color=SLIDER_COLOR)
        self.place_at_grid(slider_label, "A4", scale_factor=0.8)
        
        # Position track in Row B to avoid collision with label in Row A
        slider_track = Line(self.grid["B2"], self.grid["B5"], color=WHITE)
        slider_handle = Dot(color=SLIDER_COLOR).move_to(self.grid["B3"])
        
        self.play(
            Create(slider_track),
            Write(slider_label),
            FadeIn(slider_handle),
            run_time=1
        )
        
        # Visualizing "changing the step size"
        self.play(slider_handle.animate.move_to(self.grid["B2"]), run_time=0.8)
        self.play(slider_handle.animate.move_to(self.grid["B4"]), run_time=0.8)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Line 4: "Tiny adjustments gradually improve Byte's accuracy."
        self.lecture[3].set_color(TANGENT_COLOR)
        # Further progress down the curve
        self.play(ball_x.animate.set_value(0.8), run_time=1.2, rate_func=smooth)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: "Each step brings us closer to the bottom."
        self.lecture[4].set_color(SLOPE_COLOR)
        
        # Final approach to the minimum
        self.play(ball_x.animate.set_value(0.3), run_time=0.8)
        self.play(ball_x.animate.set_value(0.0), run_time=0.8)
        
        self.wait(2)
        # Cleanup
        self.play(
            *[FadeOut(m) for m in [curve, ball, tangent_arrow, slider_track, slider_handle, slider_label]]
        )
        self.wait(1)
