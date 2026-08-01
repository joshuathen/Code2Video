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
        # Section ID: section_4
        # Title and Lecture Lines from Storyboard
        title_text = "From Secant to Tangent: The Limit Visualized"
        lecture_lines = [
            "We start with two points on this parabola.",
            "A secant line connects them to show average change.",
            "Watch as we slide the second point closer.",
            "The gap shrinks until it is nearly zero.",
            "The secant transforms into a single-point tangent line."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_PARABOLA = "#FF00FF" # Purple
        COLOR_SECANT = "#0000FF"   # Blue
        COLOR_TANGENT = RED        # Red
        COLOR_POINT_A = WHITE
        COLOR_POINT_B = WHITE

        # 1. Setup Axes and Parabola
        # Issue 37: Move axes to B1-F6
        axes = Axes(
            x_range=[-0.5, 3, 1],
            y_range=[-0.5, 5, 1],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": True, "color": GREY}
        ).set_opacity(0.6)
        self.place_in_area(axes, "B1", "F6", scale_factor=0.9)

        # f(x) = x^2
        def func(x):
            return x**2
        
        parabola = axes.plot(func, x_range=[0, 2.2], color=COLOR_PARABOLA)
        
        # Asset integration (Issue 27)
        par_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/par.svg")
        self.place_at_grid(par_icon, "A6", scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        # "We start with two points on this parabola."
        self.lecture[0].set_color(COLOR_PARABOLA)
        
        x_a = 0.8
        x_b_start = 2.0
        
        # Points
        point_a_coords = axes.c2p(x_a, func(x_a))
        point_b_coords = axes.c2p(x_b_start, func(x_b_start))
        
        dot_a = Dot(point_a_coords, color=COLOR_POINT_A, radius=0.08)
        dot_b = Dot(point_b_coords, color=COLOR_POINT_B, radius=0.08)
        
        # Labels (Issue 38 & 39)
        # Using specific grid positions to avoid overlap
        label_a = Text("A", font_size=20)
        label_b = Text("B", font_size=20)
        self.place_at_grid(label_a, "E2", scale_factor=0.6)
        self.place_at_grid(label_b, "D3", scale_factor=0.6)

        self.play(
            Create(axes),
            Create(parabola),
            FadeIn(par_icon)
        )
        self.play(
            FadeIn(dot_a, label_a),
            FadeIn(dot_b, label_b)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "A secant line connects them to show average change."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_SECANT)

        # Calculate slope for secant line (initial)
        slope_init = (func(x_b_start) - func(x_a)) / (x_b_start - x_a)
        
        # Create a line mobject that we can update
        secant_line = Line(
            axes.c2p(x_a - 0.5, func(x_a) - 0.5 * slope_init),
            axes.c2p(x_b_start + 0.5, func(x_b_start) + 0.5 * slope_init),
            color=COLOR_SECANT,
            stroke_width=4
        )
        
        self.play(Create(secant_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Watch as we slide the second point closer."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Tracker for point B position
        x_b_tracker = ValueTracker(x_b_start)

        # Update secant line geometry
        def update_secant(line):
            curr_x_b = x_b_tracker.get_value()
            if abs(curr_x_b - x_a) < 0.0001:
                slope = 2 * x_a # Derivative of x^2 is 2x
            else:
                slope = (func(curr_x_b) - func(x_a)) / (curr_x_b - x_a)
            
            # Extend line
            start_x = x_a - 0.8
            end_x = curr_x_b + 0.5
            
            line.set_points_by_components(
                start=axes.c2p(start_x, func(x_a) + slope * (start_x - x_a)),
                end=axes.c2p(end_x, func(x_a) + slope * (end_x - x_a))
            )

        dot_b.add_updater(lambda d: d.move_to(axes.c2p(x_b_tracker.get_value(), func(x_b_tracker.get_value()))))
        secant_line.add_updater(update_secant)

        # Move B halfway
        mid_x_b = (x_b_start + x_a) / 2 + 0.2
        self.play(
            x_b_tracker.animate.set_value(mid_x_b),
            run_time=2,
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # "The gap shrinks until it is nearly zero."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Move B to nearly A
        self.play(
            x_b_tracker.animate.set_value(x_a + 0.01),
            run_time=3,
            rate_func=slow_into
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "The secant transforms into a single-point tangent line."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_TANGENT)

        # Remove updaters to transform properly
        secant_line.clear_updaters()
        dot_b.clear_updaters()
        
        # Define Tangent Line
        slope_tangent = 2 * x_a
        tangent_line = Line(
            axes.c2p(x_a - 1.2, func(x_a) - 1.2 * slope_tangent),
            axes.c2p(x_a + 1.2, func(x_a) + 1.2 * slope_tangent),
            color=COLOR_TANGENT,
            stroke_width=5
        )

        self.play(
            ReplacementTransform(secant_line, tangent_line),
            FadeOut(dot_b, label_b),
            FadeOut(label_a),
            Indicate(tangent_line, color=COLOR_TANGENT)
        )
        
        self.wait(2)

        # Reset colors
        self.play(self.lecture[4].animate.set_color(WHITE))
