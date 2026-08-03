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
        self.setup_layout(
            "Prerequisite: The Generalized Pythagorean Theorem",
            [
                "Distance in 2D follows the standard Pythagorean theorem.",
                "Each new dimension adds a perpendicular squared term.",
                "The N-dimensional sphere radius stays constant throughout."
            ]
        )

        # Colors
        color_2d = BLUE_A
        color_3d = "#FFFF00"
        color_nd = GREEN_A

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_2d)
        
        # Equation 2D: x1^2 + x2^2 = R^2
        eq_2d = MathTex("x_1^2", "+", "x_2^2", "=", "R^2", font_size=40)
        # Fix Issue 33: Change area to B1-C6 and scale_factor to 0.8
        self.place_in_area(eq_2d, "B1", "C6", scale_factor=0.8)
        
        # 2D Geometry
        origin_pt = ORIGIN
        axis_x1 = Arrow(origin_pt, origin_pt + RIGHT * 1.8, buff=0, color=WHITE)
        axis_x2 = Arrow(origin_pt, origin_pt + UP * 1.8, buff=0, color=WHITE)
        label_x1 = MathTex("x_1", font_size=24).next_to(axis_x1, RIGHT, buff=0.1)
        label_x2 = MathTex("x_2", font_size=24).next_to(axis_x2, UP, buff=0.1)
        
        # Radius vector in 2D
        vec_2d_end = origin_pt + RIGHT * 1.2 + UP * 1.0
        radius_vec = Line(origin_pt, vec_2d_end, color=color_2d)
        radius_dot = Dot(vec_2d_end, color=color_2d, radius=0.06)
        radius_label = MathTex("R", font_size=24, color=color_2d).next_to(radius_vec.get_center(), UL, buff=0.05)
        
        geometry_2d = VGroup(axis_x1, axis_x2, label_x1, label_x2, radius_vec, radius_dot, radius_label)
        # Fix Issue 34: Change scale_factor to 0.75
        self.place_in_area(geometry_2d, "D1", "F6", scale_factor=0.75)
        
        self.play(Write(eq_2d), FadeIn(geometry_2d))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(color_3d)
        
        # Add 3rd axis
        # geom_origin is where axis_x1 starts (already placed by place_in_area)
        geom_origin = axis_x1.get_start()
        axis_x3 = Arrow(geom_origin, geom_origin + DL * 1.2, buff=0, color=WHITE)
        label_x3 = MathTex("x_3", font_size=24).next_to(axis_x3, DL, buff=0.1)
        
        # Equation 3D: x1^2 + x2^2 + x3^2 = R^2
        eq_3d = MathTex("x_1^2", "+", "x_2^2", "+", "x_3^2", "=", "R^2", font_size=40)
        eq_3d.set_color_by_tex("x_3^2", color_3d)
        # Fix Issue 33: Change area to B1-C6 and scale_factor to 0.8
        self.place_in_area(eq_3d, "B1", "C6", scale_factor=0.8)
        
        self.play(
            Transform(eq_2d, eq_3d),
            Create(axis_x3),
            Write(label_x3)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(color_nd)
        
        # Generalize to N dimensions
        eq_nd = MathTex("x_1^2", "+", "\\dots", "+", "x_n^2", "=", "R^2", font_size=40)
        # Fix Issue 33: Change area to B1-C6 and scale_factor to 0.8
        self.place_in_area(eq_nd, "B1", "C6", scale_factor=0.8)
        
        # Identify the ellipsis part for pulsing
        ellipsis = eq_nd[2]
        
        self.play(Transform(eq_2d, eq_nd))
        
        # Pulsing effect for the ellipsis using an updater
        pulse_tracker = ValueTracker(0)
        def pulse_updater(m, dt):
            pulse_tracker.increment_value(dt)
            t = pulse_tracker.get_value()
            m.set_opacity(0.4 + 0.6 * np.abs(np.sin(t * 3)))
            
        ellipsis.add_updater(pulse_updater)
        
        # Add visual hint for N-dimensions (higher dimensional points)
        dots_nd = VGroup(*[
            Dot(
                radius_dot.get_center() + np.array([
                    np.random.uniform(-0.3, 0.3), 
                    np.random.uniform(-0.3, 0.3), 
                    0
                ]), 
                radius=0.03, 
                color=color_nd
            ) for _ in range(8)
        ])
        
        self.play(FadeIn(dots_nd))
        self.wait(4)
        
        ellipsis.remove_updater(pulse_updater)
        self.wait(2)
