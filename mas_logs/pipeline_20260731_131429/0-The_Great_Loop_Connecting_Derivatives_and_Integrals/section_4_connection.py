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
        # L1: Show a 3D printer nozzle (#C0C0C0) moving across a layer. Label its speed as 'Derivative' (#FF8C00).
        self.lecture[0].set_color(YELLOW)
        
        nozzle = Triangle(color=COLOR_NOZZLE, fill_opacity=1.0).scale(0.2).rotate(180*DEGREES)
        nozzle_label = Text("Derivative (Speed)", font_size=16, color=COLOR_DERIVATIVE)
        nozzle_vg = VGroup(nozzle, nozzle_label).arrange(DOWN, buff=0.1)
        # Fix Issue 31: nozzle_vg at B2 obscured by later graphic. Move to A3.
        self.place_at_grid(nozzle_vg, "A3", scale_factor=0.7)
        
        # Adjust path to match row A
        path = Line(self.grid["A2"], self.grid["A6"], color=GREY_E)
        self.add(path)
        
        self.play(FadeIn(nozzle_vg))
        self.play(nozzle_vg.animate.move_to(self.grid["A6"]), run_time=1.5, rate_func=linear)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # L2: Show the finished model (#32CD32) growing layer by layer. Label the total model as 'Integral' (#00FA9A).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        model_base = Line(self.grid["E1"], self.grid["E5"], color=GREY_B)
        # Create a rectangle that starts small and grows
        model_rect = Rectangle(width=4.0, height=0.01, color=COLOR_MODEL, fill_opacity=0.8, stroke_width=0)
        model_rect.move_to(model_base.get_center(), aligned_edge=DOWN)
        model_label = Text("Integral (Model)", font_size=16, color=COLOR_INTEGRAL)
        # Fix Issue 32: model_label at F3 is too far. Move to E3.
        self.place_at_grid(model_label, "E3", scale_factor=0.8)
        
        self.play(Create(model_base), FadeIn(model_label))
        self.play(model_rect.animate.stretch_to_fit_height(1.5).move_to(model_base.get_center(), aligned_edge=DOWN), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # L3: Display a 'Distance' graph (#FFFFFF) and a 'Speed' graph (#FFD700) stacked vertically.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        self.play(FadeOut(nozzle_vg, path, model_base, model_rect, model_label))

        dist_axes = Axes(
            x_range=[0, 4, 1], y_range=[0, 4, 1],
            x_length=3, y_length=1.5,
            axis_config={"color": WHITE, "include_tip": False}
        ).scale(0.8)
        dist_label = Text("Distance", font_size=14, color=COLOR_DISTANCE)
        dist_vg = VGroup(dist_axes, dist_label).arrange(UP, buff=0.1)
        self.place_in_area(dist_vg, "A4", "B6")

        speed_axes = Axes(
            x_range=[0, 4, 1], y_range=[0, 4, 1],
            x_length=3, y_length=1.5,
            axis_config={"color": COLOR_SPEED, "include_tip": False}
        ).scale(0.8)
        speed_label = Text("Speed", font_size=14, color=COLOR_SPEED)
        speed_vg = VGroup(speed_axes, speed_label).arrange(UP, buff=0.1)
        self.place_in_area(speed_vg, "D4", "E6")

        dist_curve = dist_axes.plot(lambda x: 0.2 * x**2, x_range=[0, 4], color=COLOR_DISTANCE)
        speed_curve = speed_axes.plot(lambda x: 0.4 * x, x_range=[0, 4], color=COLOR_SPEED)

        self.play(Create(dist_axes), Create(speed_axes), FadeIn(dist_label), FadeIn(speed_label))
        self.play(Create(dist_curve), Create(speed_curve))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # L4: Animate a point on the Speed graph; show the corresponding Slope of the Distance graph changing.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        vt = ValueTracker(0.1)
        
        speed_dot = Dot(color=COLOR_SPEED, radius=0.08)
        speed_dot.add_updater(lambda d: d.move_to(speed_axes.c2p(vt.get_value(), 0.4 * vt.get_value())))
        
        tangent = Line(LEFT, RIGHT, color=COLOR_DERIVATIVE).scale(0.4)
        def update_tangent(t):
            x = vt.get_value()
            slope = 0.4 * x
            dx = 0.3
            # Use points for robust tangent line update
            p1 = dist_axes.c2p(x - dx, 0.2 * x**2 - slope * dx)
            p2 = dist_axes.c2p(x + dx, 0.2 * x**2 + slope * dx)
            t.set_points_as_corners([p1, p2])

        tangent.add_updater(update_tangent)
        
        self.add(speed_dot, tangent)
        self.play(vt.animate.set_value(3.8), run_time=2.5, rate_func=linear)
        tangent.clear_updaters()
        speed_dot.clear_updaters()
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # L5: This link is the Fundamental Theorem of Calculus.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Fade out graphs slightly to highlight the FTC symbols
        self.play(VGroup(dist_vg, speed_vg, dist_curve, speed_curve, tangent, speed_dot).animate.set_opacity(0.3))

        deriv_sym = Text("d/dx", color=COLOR_DERIVATIVE, font_size=32)
        integ_sym = Text("∫", color=COLOR_INTEGRAL, font_size=40)
        
        # Large 'UNDO' button icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/no.svg]
        undo_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/no.svg").set_color(COLOR_UNDO)
        undo_label = Text("UNDO", font_size=14, color=COLOR_UNDO)
        undo_btn = VGroup(undo_icon, undo_label).arrange(DOWN, buff=0.1)
        
        ftc_vg = VGroup(deriv_sym, undo_btn, integ_sym).arrange(RIGHT, buff=0.5)
        # Fix Issue 30: ftc_vg overlapping lecture notes. Move to B2-D3.
        self.place_in_area(ftc_vg, "B2", "D3", scale_factor=0.6)
        
        ftc_title = Text("Fundamental Theorem", font_size=20, color=YELLOW)
        self.place_at_grid(ftc_title, "A2")
        
        self.play(FadeIn(ftc_title), Write(deriv_sym), Write(integ_sym))
        self.play(FadeIn(undo_btn))
        self.play(Indicate(undo_btn, scale_factor=1.2))
        self.wait(2)
