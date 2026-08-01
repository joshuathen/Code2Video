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
        title = "Trigonometric Intuition: The Unit Circle Dance"
        lines = [
            "Imagine a point moving around a unit circle.",
            "The height of the point represents the sine function.",
            "Its vertical speed depends on its horizontal position.",
            "When height changes fastest, horizontal value is highest.",
            "This proves the derivative of sine is cosine."
        ]
        self.setup_layout(title, lines)

        # Assets/Values
        theta = ValueTracker(0.0)
        radius = 1.5
        
        # Colors for matching lecture lines
        COLOR_SIN = "#0000FF"
        COLOR_COS = "#FF0000"
        COLOR_FASTEST = "#00FFFF"
        
        # Grid center for the unit circle (placed to leave room for formula below)
        circle_center = np.array([3.0, 0.2, 0])

        # === Animation for Lecture Line 1 ===
        # Imagine a point moving around a unit circle.
        unit_circle = Circle(radius=radius, color=WHITE)
        unit_circle.move_to(circle_center)
        
        # Integration of Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/point.svg
        point_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/point.svg").set_color(WHITE).scale(0.2)
        point_asset.add_updater(lambda d: d.move_to(
            circle_center + np.array([radius * np.cos(theta.get_value()), radius * np.sin(theta.get_value()), 0])
        ))
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            Create(unit_circle),
            FadeIn(point_asset)
        )
        self.play(theta.animate.set_value(0.5), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The height of the point represents the sine function.
        blue_line = Line(color=COLOR_SIN)
        blue_line.add_updater(lambda l: l.set_points_as_corners([
            circle_center + np.array([radius * np.cos(theta.get_value()), 0, 0]),
            circle_center + np.array([radius * np.cos(theta.get_value()), radius * np.sin(theta.get_value()), 0])
        ]))
        
        sin_label = MathTex(r"\sin(x)", color=COLOR_SIN, font_size=24)
        sin_label.add_updater(lambda l: l.next_to(blue_line, RIGHT, buff=0.1))
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_SIN),
            Create(blue_line),
            FadeIn(sin_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Its vertical speed depends on its horizontal position.
        red_line = Line(color=COLOR_COS)
        red_line.add_updater(lambda l: l.set_points_as_corners([
            circle_center,
            circle_center + np.array([radius * np.cos(theta.get_value()), 0, 0])
        ]))
        
        cos_label = MathTex(r"\cos(x)", color=COLOR_COS, font_size=24)
        cos_label.add_updater(lambda l: l.next_to(red_line, DOWN, buff=0.1))

        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_COS),
            Create(red_line),
            FadeIn(cos_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # When height changes fastest, horizontal value is highest.
        def get_flash_anim(pos, color):
            f = Circle(radius=0.1, color=color, stroke_width=6).move_to(pos)
            return AnimationGroup(
                f.animate.scale(6).set_stroke(opacity=0),
                run_time=0.6,
                rate_func=linear
            )

        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_FASTEST)
        )
        
        # Point at 0: Fast (Cyan flash)
        self.play(theta.animate.set_value(0), run_time=0.5)
        self.play(get_flash_anim(circle_center + np.array([radius, 0, 0]), COLOR_FASTEST))
        
        # To pi/2: Stop (Red flash at center because cos=0)
        self.play(theta.animate.set_value(PI/2), run_time=1.5, rate_func=linear)
        self.play(get_flash_anim(circle_center, COLOR_COS))
        
        # To pi: Fast (Cyan flash)
        self.play(theta.animate.set_value(PI), run_time=1.5, rate_func=linear)
        self.play(get_flash_anim(circle_center + np.array([-radius, 0, 0]), COLOR_FASTEST))
        
        # To 3pi/2: Stop (Red flash)
        self.play(theta.animate.set_value(3*PI/2), run_time=1.5, rate_func=linear)
        self.play(get_flash_anim(circle_center, COLOR_COS))

        # Back to 2pi
        self.play(theta.animate.set_value(2*PI), run_time=1.5, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This proves the derivative of sine is cosine.
        formula = MathTex(r"\frac{d}{dx}(\sin x) = \cos x", color=WHITE)
        # Positioned below the circle to avoid overlap (Issue 35)
        self.place_in_area(formula, 'E2', 'F5', scale_factor=0.8)
        
        bg_rect = SurroundingRectangle(formula, color=BLACK, fill_opacity=0.7, stroke_width=0)

        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(WHITE),
            FadeIn(bg_rect),
            Write(formula)
        )
        self.wait(3)
