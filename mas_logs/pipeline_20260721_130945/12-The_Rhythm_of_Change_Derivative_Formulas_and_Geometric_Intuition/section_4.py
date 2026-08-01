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

        # Colors based on storyboard and common conventions
        COLOR_PARABOLA = "#FF00FF" # Purple
        COLOR_SECANT = "#0000FF"   # Blue
        COLOR_TANGENT = RED        # Red
        COLOR_POINT = WHITE
        COLOR_HIGHLIGHT = YELLOW

        # 1. Setup Axes and Parabola
        # Issue 37: Move axes to B1-F6 to avoid crowding lecture notes
        axes = Axes(
            x_range=[-0.2, 2.5, 1],
            y_range=[-0.5, 5, 1],
            x_length=5,
            y_length=5,
            axis_config={"include_tip": True, "color": GREY}
        )
        self.place_in_area(axes, "B1", "F6", scale_factor=0.9)

        # f(x) = x^2
        def func(x):
            return x**2
        
        parabola = axes.plot(func, x_range=[0, 2.2], color=COLOR_PARABOLA)
        
        # Issue 27: Asset integration - Parabola icon
        par_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/par.svg")
        self.place_at_grid(par_icon, "A6", scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        # "We start with two points on this parabola."
        x_a = 0.8
        x_b_start = 2.0
        
        dot_a = Dot(axes.c2p(x_a, func(x_a)), color=COLOR_POINT, radius=0.08)
        dot_b = Dot(axes.c2p(x_b_start, func(x_b_start)), color=COLOR_POINT, radius=0.08)
        
        # Issue 38 & 39: Labels A and B (referencing P and Q from issues for grid positioning)
        label_a = Text("A", font_size=24)
        label_b = Text("B", font_size=24)
        self.place_at_grid(label_a, "E2", scale_factor=0.6)
        self.place_at_grid(label_b, "D3", scale_factor=0.6)

        self.play(
            self.lecture[0].animate.set_color(COLOR_PARABOLA),
            Create(axes),
            Create(parabola),
            FadeIn(par_icon),
            run_time=1.5
        )
        self.play(
            FadeIn(dot_a, label_a),
            FadeIn(dot_b, label_b),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "A secant line connects them to show average change."
        slope_init = (func(x_b_start) - func(x_a)) / (x_b_start - x_a)
        secant_line = Line(
            axes.c2p(x_a - 0.5, func(x_a) - 0.5 * slope_init),
            axes.c2p(x_b_start + 0.5, func(x_b_start) + 0.5 * slope_init),
            color=COLOR_SECANT,
            stroke_width=4
        )
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_SECANT),
            Create(secant_line),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Watch as we slide the second point closer."
        x_b_tracker = ValueTracker(x_b_start)

        # Optimization: Simplified updater using put_start_and_end_on to ensure render stability
        def update_secant(line):
            curr_x_b = x_b_tracker.get_value()
            if abs(curr_x_b - x_a) < 0.001:
                slope = 2 * x_a
            else:
                slope = (func(curr_x_b) - func(x_a)) / (curr_x_b - x_a)
            
            p1 = axes.c2p(x_a - 0.7, func(x_a) - 0.7 * slope)
            p2 = axes.c2p(curr_x_b + 0.7, func(curr_x_b) + 0.7 * slope)
            line.put_start_and_end_on(p1, p2)

        dot_b.add_updater(lambda d: d.move_to(axes.c2p(x_b_tracker.get_value(), func(x_b_tracker.get_value()))))
        secant_line.add_updater(update_secant)

        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT),
            x_b_tracker.animate.set_value(1.4),
            run_time=2,
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # "The gap shrinks until it is nearly zero."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_HIGHLIGHT),
            x_b_tracker.animate.set_value(x_a + 0.05),
            run_time=2,
            rate_func=slow_into
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "The secant transforms into a single-point tangent line."
        # Clear updaters before transform for performance and correctness
        secant_line.clear_updaters()
        dot_b.clear_updaters()
        
        slope_tangent = 2 * x_a
        tangent_line = Line(
            axes.c2p(x_a - 1.2, func(x_a) - 1.2 * slope_tangent),
            axes.c2p(x_a + 1.2, func(x_a) + 1.2 * slope_tangent),
            color=COLOR_TANGENT,
            stroke_width=5
        )

        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(COLOR_TANGENT),
            ReplacementTransform(secant_line, tangent_line),
            FadeOut(dot_b, label_b),
            Indicate(tangent_line, color=COLOR_TANGENT),
            run_time=1.5
        )
        self.wait(2)

        # Reset final color
        self.play(self.lecture[4].animate.set_color(WHITE), run_time=1)
