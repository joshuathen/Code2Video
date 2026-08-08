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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        title = "Prerequisite: The Rule of Distance"
        lines = [
            "A sphere is points at fixed distance from center.",
            "In 2D, the Pythagorean Theorem defines a circle.",
            "Higher dimensions just add more squared coordinate terms."
        ]
        self.setup_layout(title, lines)
        
        # Colors
        COLOR_2D = "#00FFFF"
        COLOR_3D = "#00FFFF"
        COLOR_R = "#FFFF00"
        
        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color(COLOR_2D))
        
        # 2D Axes setup
        # Optimized grid area per VideoCritic feedback (Issue 39)
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": GREY}
        )
        self.place_in_area(axes, "B1", "E6")
        
        center_dot = Dot(axes.c2p(0, 0), color=WHITE, radius=0.05)
        point = Dot(axes.c2p(1.5, 0), color=COLOR_2D)
        radius_line = Line(axes.c2p(0, 0), axes.c2p(1.5, 0), color=COLOR_R)
        label_r = MathTex("R", color=COLOR_R, font_size=24)
        label_r.next_to(radius_line, UP, buff=0.1)
        
        self.play(Create(axes), FadeIn(center_dot))
        self.play(Create(radius_line), FadeIn(point), Write(label_r))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Transition lecture line focus
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_2D)
        )
        
        # Tracing a circle
        circle = Circle(radius=1.5, color=COLOR_2D).move_to(axes.c2p(0, 0))
        angle_tracker = ValueTracker(0)
        
        # Updaters for the tracing point and radius
        point.add_updater(lambda d: d.move_to(axes.c2p(1.5 * np.cos(angle_tracker.get_value()), 1.5 * np.sin(angle_tracker.get_value()))))
        radius_line.add_updater(lambda l: l.become(Line(axes.c2p(0, 0), point.get_center(), color=COLOR_R)))
        label_r.add_updater(lambda m: m.next_to(radius_line, UP if abs(np.sin(angle_tracker.get_value())) < 0.7 else RIGHT, buff=0.1))
        
        self.play(Create(circle), angle_tracker.animate.set_value(2 * PI), run_time=3, rate_func=linear)
        self.wait(0.5)
        
        # Clean up updaters
        point.clear_updaters()
        radius_line.clear_updaters()
        label_r.clear_updaters()
        
        # Display 2D Formula
        # Expanded formula area per VideoCritic feedback (Issue 40)
        formula_2d = MathTex("x^2 + y^2 = R^2", color=COLOR_2D)
        self.place_in_area(formula_2d, "A1", "A6", scale_factor=1.0)
        self.play(Write(formula_2d))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Transition lecture line focus
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_3D)
        )
        
        # Reveal 3rd dimension (Z-axis)
        z_axis_end = axes.c2p(0, 0) + np.array([-1.2, -1.2, 0])
        z_axis = Line(axes.c2p(0, 0), z_axis_end, color=GREY).add_tip()
        z_label = MathTex("z", color=GREY, font_size=24).next_to(z_axis_end, DOWN+LEFT, buff=0.1)
        
        # Integrate sphere asset per Issue 20
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg]
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_color(COLOR_3D)
        self.place_in_area(sphere_asset, "B1", "E6", scale_factor=1.3)
        
        self.play(
            Create(z_axis),
            FadeIn(z_label),
            ReplacementTransform(circle, sphere_asset),
            FadeOut(point),
            FadeOut(radius_line),
            FadeOut(label_r)
        )
        self.wait(1)
        
        # Update formula to 3D
        formula_3d = MathTex("x^2 + y^2 + z^2 = R^2", color=COLOR_3D)
        self.place_in_area(formula_3d, "A1", "A6", scale_factor=1.0)
        self.play(ReplacementTransform(formula_2d, formula_3d))
        self.wait(1)
        
        # Transition to General n-Dimensions
        formula_nd = MathTex("x_1^2 + x_2^2 + \\dots + x_n^2 = R^2", color=COLOR_3D)
        self.place_in_area(formula_nd, "A1", "A6", scale_factor=1.0)
        self.play(ReplacementTransform(formula_3d, formula_nd))
        self.wait(1)
        
        # Highlight 'R' as a constant
        self.play(formula_nd.animate.set_color_by_tex("R", COLOR_R))
        
        # Corrected explanation placement per VideoCritic feedback (Issue 38)
        explanation = Text("R is a constant distance", font_size=22, color=COLOR_R)
        self.place_in_area(explanation, "F2", "F5", scale_factor=0.8)
        self.play(Write(explanation))
        self.wait(2)
