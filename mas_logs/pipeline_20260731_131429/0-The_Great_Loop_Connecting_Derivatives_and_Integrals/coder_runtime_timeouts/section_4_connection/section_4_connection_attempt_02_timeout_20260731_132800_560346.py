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
        deriv_sym = Text("d/dx", color=COLOR_DERIVATIVE, font_size=36)
        integ_sym = Text("∫", color=COLOR_INTEGRAL, font_size=48)
        
        self.place_at_grid(deriv_sym, "A2", scale_factor=1.0)
        self.place_at_grid(integ_sym, "C2", scale_factor=1.0)
        
        self.play(FadeIn(deriv_sym), FadeIn(integ_sym))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Use Asset for UNDO button
        undo_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/no.svg").set_color(COLOR_UNDO)
        undo_label = Text("UNDO", font_size=16, color=COLOR_UNDO)
        undo_button = VGroup(undo_asset, undo_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(undo_button, "B2", scale_factor=0.5)
        
        plus_minus = Text("+ ↔ -", font_size=24, color=WHITE)
        self.place_at_grid(plus_minus, "D2", scale_factor=1.0)
        
        self.play(FadeIn(undo_button))
        self.play(Write(plus_minus))
        self.wait(0.5)
        self.play(FadeOut(plus_minus))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Fade symbols to background
        self.play(VGroup(deriv_sym, integ_sym, undo_button).animate.set_opacity(0.3))

        # Graphs setup
        dist_axes = Axes(
            x_range=[0, 4, 1], y_range=[0, 4, 1],
            x_length=2.0, y_length=1.2,
            axis_config={"include_tip": False, "color": COLOR_DISTANCE}
        )
        dist_label = Text("Distance", font_size=14, color=COLOR_DISTANCE)
        dist_vg = VGroup(dist_axes, dist_label).arrange(UP, buff=0.1)
        self.place_in_area(dist_vg, "A4", "B6", scale_factor=1.0)
        
        speed_axes = Axes(
            x_range=[0, 4, 1], y_range=[0, 4, 1],
            x_length=2.0, y_length=1.2,
            axis_config={"include_tip": False, "color": COLOR_SPEED}
        )
        speed_label = Text("Speed", font_size=14, color=COLOR_SPEED)
        speed_vg = VGroup(speed_axes, speed_label).arrange(UP, buff=0.1)
        self.place_in_area(speed_vg, "D4", "E6", scale_factor=1.0)

        dist_curve = dist_axes.plot(lambda x: x**2 / 4, x_range=[0, 4], color=COLOR_DISTANCE)
        speed_curve = speed_axes.plot(lambda x: x / 2, x_range=[0, 4], color=COLOR_SPEED)

        # 3D Printer Nozzle visual
        nozzle = Triangle(color=COLOR_NOZZLE, fill_opacity=1.0).scale(0.15).rotate(180*DEGREES)
        nozzle_label = Text("Nozzle", font_size=12, color=COLOR_DERIVATIVE)
        nozzle_vg = VGroup(nozzle, nozzle_label).arrange(DOWN, buff=0.05)
        self.place_at_grid(nozzle_vg, "E1", scale_factor=1.0)

        self.play(Create(dist_axes), Create(speed_axes), FadeIn(dist_label), FadeIn(speed_label))
        self.play(Create(dist_curve), FadeIn(nozzle_vg))
        
        # Syncing animation with ValueTracker
        vt = ValueTracker(0)
        
        # Derivative visual: tangent line segment
        tangent = Line(LEFT, RIGHT, color=COLOR_DERIVATIVE, stroke_width=4).scale(0.2)
        tangent.add_updater(lambda m: m.move_to(dist_axes.c2p(vt.get_value(), vt.get_value()**2/4)).set_angle(np.arctan(vt.get_value()/2)))
        
        # Dot on speed graph
        speed_dot = Dot(color=COLOR_SPEED, radius=0.06)
        speed_dot.add_updater(lambda m: m.move_to(speed_axes.c2p(vt.get_value(), vt.get_value()/2)))

        self.add(tangent, speed_dot)
        self.play(
            vt.animate.set_value(4),
            nozzle_vg.animate.shift(RIGHT*2),
            Create(speed_curve),
            run_time=3,
            rate_func=linear
        )
        tangent.clear_updaters()
        speed_dot.clear_updaters()
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Finished Model (Integral)
        model_base = Line(LEFT, RIGHT, color=GREY).scale(0.5)
        self.place_at_grid(model_base, "F2", scale_factor=1.0)
        
        # Use a rectangle that grows
        model_rect = Rectangle(height=0.01, width=0.8, color=COLOR_MODEL, fill_opacity=0.8, stroke_width=1)
        model_rect.move_to(model_base.get_center(), aligned_edge=DOWN)
        
        model_label = Text("Finished Model", font_size=14, color=COLOR_MODEL)
        self.place_at_grid(model_label, "F3", scale_factor=1.0)
        
        # Area under speed graph representing integral
        area = speed_axes.get_area(speed_curve, x_range=[0, 4], color=COLOR_INTEGRAL, opacity=0.3)
        
        self.play(Create(model_base), FadeIn(model_label))
        
        # Growth animation for model
        vt.set_value(0)
        model_rect.add_updater(lambda m: m.set_height(max(0.01, vt.get_value() * 0.3), stretch=True).move_to(model_base.get_center(), aligned_edge=DOWN))
        self.add(model_rect)
        
        self.play(
            Create(area),
            vt.animate.set_value(4),
            run_time=2
        )
        model_rect.clear_updaters()
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        ftc_text = Text("Fundamental Theorem of Calculus", font_size=20, color=YELLOW)
        self.place_in_area(ftc_text, "B1", "C3", scale_factor=1.0)
        ftc_box = SurroundingRectangle(ftc_text, color=YELLOW, buff=0.1)
        
        self.play(
            FadeIn(ftc_box), 
            Write(ftc_text),
            VGroup(deriv_sym, integ_sym, undo_button).animate.set_opacity(1).scale(1.2)
        )
        self.play(Indicate(undo_button, color=COLOR_UNDO))
        self.wait(2)
