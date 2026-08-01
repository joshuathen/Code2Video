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

class Section4ConnectionScene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Differentiation and integration are inverse operations.",
            "One \"undoes\" the other, like adding and subtracting.",
            "The derivative of distance is your current speed.",
            "The integral of speed is your total distance.",
            "This link is the Fundamental Theorem of Calculus."
        ]
        self.setup_layout("The Fundamental Theorem: The 'Undo' Button", lecture_lines)
        
        # Colors
        COLOR_DERIVATIVE = "#FF8C00"
        COLOR_INTEGRAL = "#00FA9A"
        COLOR_NOZZLE = "#C0C0C0"
        COLOR_MODEL = "#32CD32"
        COLOR_DISTANCE = "#FFFFFF"
        COLOR_SPEED = "#FFD700"
        COLOR_UNDO = "#FF0000"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        deriv_sym = MathTex(r"\frac{d}{dx}", color=COLOR_DERIVATIVE)
        integ_sym = MathTex(r"\int", color=COLOR_INTEGRAL)
        
        self.place_at_grid(deriv_sym, "A2", scale_factor=1.2)
        self.place_at_grid(integ_sym, "C2", scale_factor=1.2)
        
        self.play(FadeIn(deriv_sym), FadeIn(integ_sym))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        undo_box = RoundedRectangle(corner_radius=0.1, height=0.6, width=1.2, color=COLOR_UNDO, fill_opacity=0.3)
        undo_text = Text("UNDO", font_size=18, color=COLOR_UNDO)
        undo_button = VGroup(undo_box, undo_text)
        self.place_at_grid(undo_button, "B2", scale_factor=1.0)
        
        plus_minus = MathTex(r"+ \longleftrightarrow -", font_size=30, color=WHITE)
        self.place_at_grid(plus_minus, "D2", scale_factor=0.8)
        
        self.play(FadeIn(undo_button))
        self.play(Write(plus_minus))
        self.wait(1)
        self.play(FadeOut(plus_minus))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Symbols stay but fade slightly
        self.play(VGroup(deriv_sym, integ_sym, undo_button).animate.set_opacity(0.3))

        # Graphs
        dist_axes = Axes(
            x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False},
            x_length=2.5, y_length=1.5
        ).set_color(COLOR_DISTANCE)
        dist_label = Text("Distance", font_size=16, color=COLOR_DISTANCE)
        dist_vg = VGroup(dist_axes, dist_label).arrange(UP, buff=0.1)
        self.place_in_area(dist_vg, "A4", "C6", scale_factor=1.0)
        
        speed_axes = Axes(
            x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_tip": False},
            x_length=2.5, y_length=1.5
        ).set_color(COLOR_SPEED)
        speed_label = Text("Speed", font_size=16, color=COLOR_SPEED)
        speed_vg = VGroup(speed_axes, speed_label).arrange(UP, buff=0.1)
        self.place_in_area(speed_vg, "D4", "F6", scale_factor=1.0)

        dist_curve = dist_axes.plot(lambda x: x**2 / 4, x_range=[0, 4], color=COLOR_DISTANCE)
        speed_curve = speed_axes.plot(lambda x: x / 2, x_range=[0, 4], color=COLOR_SPEED)

        # 3D Printer Nozzle
        nozzle = Triangle(color=COLOR_NOZZLE, fill_opacity=1.0).scale(0.2).rotate(180*DEGREES)
        nozzle_label = Text("Speed (Deriv.)", font_size=16, color=COLOR_DERIVATIVE)
        nozzle_vg = VGroup(nozzle, nozzle_label).arrange(UP, buff=0.1)
        self.place_at_grid(nozzle_vg, "E1", scale_factor=0.8)

        self.play(Create(dist_axes), Create(speed_axes), FadeIn(dist_label), FadeIn(speed_label))
        self.play(Create(dist_curve), FadeIn(nozzle_vg))
        
        # Derivative linking
        tracker = ValueTracker(0.01)
        
        # Use simple lines and dots for updaters
        slope_line = Line(color=COLOR_DERIVATIVE, stroke_width=4)
        def update_slope_line(line):
            val = tracker.get_value()
            slope = val / 2 # Derivative of x^2/4 is x/2
            p = dist_axes.c2p(val, val**2 / 4)
            # Tangent line segment
            line.set_points_as_corners([p + LEFT*0.5 + DOWN*0.5*slope, p + RIGHT*0.5 + UP*0.5*slope])

        slope_line.add_updater(update_slope_line)
        
        speed_dot = Dot(color=COLOR_SPEED)
        speed_dot.add_updater(lambda d: d.move_to(speed_axes.c2p(tracker.get_value(), tracker.get_value()/2)))
        
        self.add(slope_line, speed_dot)
        self.play(
            tracker.animate.set_value(4), 
            Create(speed_curve), 
            nozzle_vg.animate.shift(RIGHT * 1.5), 
            run_time=4, 
            rate_func=linear
        )
        slope_line.clear_updaters()
        speed_dot.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Finished Model (Integral)
        model_base = Line(LEFT, RIGHT, color=GREY).scale(0.5)
        self.place_at_grid(model_base, "F2", scale_factor=1.0)
        
        model_rect = Rectangle(height=0.01, width=1.0, color=COLOR_MODEL, fill_opacity=0.8)
        model_rect.move_to(model_base.get_center(), aligned_edge=DOWN)
        
        model_label = Text("Model (Integral)", font_size=16, color=COLOR_INTEGRAL)
        self.place_at_grid(model_label, "F3", scale_factor=0.8)
        
        # Area under speed graph
        area = speed_axes.get_area(speed_curve, x_range=[0, 4], color=COLOR_INTEGRAL, opacity=0.3)
        
        self.play(Create(model_base), FadeIn(model_label), Create(model_rect))
        
        # Growth animation
        tracker.set_value(0.01)
        model_rect.add_updater(lambda m: m.set_height(max(0.01, tracker.get_value() * 0.4), stretch=True).move_to(model_base.get_center(), aligned_edge=DOWN))
        
        self.play(Create(area), tracker.animate.set_value(4), run_time=3)
        model_rect.clear_updaters()
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        ftc_text = Text("Fundamental Theorem of Calculus", font_size=24, color=YELLOW)
        self.place_in_area(ftc_text, "B2", "C5", scale_factor=0.9)
        ftc_box = SurroundingRectangle(ftc_text, color=YELLOW, buff=0.2)
        
        # Highlight everything
        self.play(
            FadeIn(ftc_box), 
            Write(ftc_text), 
            VGroup(deriv_sym, integ_sym, undo_button).animate.set_opacity(1).scale(1.2)
        )
        self.play(Indicate(undo_button, color=COLOR_UNDO), Indicate(ftc_text))
        self.wait(3)
