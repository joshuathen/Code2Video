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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup layout with title and lecture lines
        self.setup_layout("The 1D Borsuk-Ulam: The Equator Example", [
            "On the equator, two opposite points share a temperature.",
            "Define a function for the temperature difference at opposites.",
            "If the difference flips sign, it must hit zero.",
            "This zero point means temperatures are exactly equal.",
            "This is the Intermediate Value Theorem in action."
        ])

        # === Animation for Lecture Line 1 ===
        line1_color = "#FFD700"
        self.play(self.lecture[0].animate.set_color(line1_color))

        # Earth globe representation
        globe_circle = Circle(radius=1.2, color=BLUE_D, fill_opacity=0.2)
        equator = Ellipse(width=2.4, height=0.6, color=line1_color, stroke_width=4)
        
        globe_group = VGroup(globe_circle, equator)
        self.place_in_area(globe_group, "A2", "C5")
        
        self.play(Create(globe_circle), Create(equator))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        line2_color = "#87CEEB"
        self.play(self.lecture[1].animate.set_color(line2_color))

        center = globe_group.get_center()
        theta = ValueTracker(0)
        
        dot1 = Dot(color=line2_color).add_updater(
            lambda d: d.move_to(center + np.array([1.2 * np.cos(theta.get_value()), 0.3 * np.sin(theta.get_value()), 0]))
        )
        dot2 = Dot(color=WHITE).add_updater(
            lambda d: d.move_to(center + np.array([1.2 * np.cos(theta.get_value() + PI), 0.3 * np.sin(theta.get_value() + PI), 0]))
        )

        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        t_label = Text("T", color=line2_color, font_size=24, slant=ITALIC).add_updater(
            lambda m: m.next_to(dot1, UP, buff=0.1)
        )
        t_prime_label = Text("T'", color=WHITE, font_size=24, slant=ITALIC).add_updater(
            lambda m: m.next_to(dot2, UP, buff=0.1)
        )

        self.add(dot1, dot2, t_label, t_prime_label)
        self.play(theta.animate.set_value(PI/2), run_time=2)

        # === Animation for Lecture Line 3 ===
        line3_color = "#FF6347"
        self.play(self.lecture[2].animate.set_color(line3_color))

        axes = Axes(
            x_range=[0, TAU, PI],
            y_range=[-1.5, 1.5, 1],
            x_length=4,
            y_length=2,
            axis_config={"color": WHITE, "include_tip": False},
            x_axis_config={"stroke_width": 2},
            y_axis_config={"stroke_width": 2}
        )
        # Fix for Issue 29: relayout axes to avoid overlap with row F
        self.place_in_area(axes, "D2", "E5", scale_factor=0.9)
        
        # Replaced MathTex with Text
        g_label = Text("g(x) = T(x) - T(-x)", font_size=20, color=line3_color, slant=ITALIC)
        # Fix for Issue 30: Place g_label in an area for better centering
        self.place_in_area(g_label, "D2", "D4", scale_factor=0.7)

        graph = axes.plot(lambda x: np.sin(x), color=line3_color)
        
        self.play(Create(axes), Write(g_label))
        self.play(Create(graph))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        line4_color = "#FFFF00"
        self.play(self.lecture[3].animate.set_color(line4_color))

        graph_dot = Dot(color=line4_color).add_updater(
            lambda d: d.move_to(axes.c2p(theta.get_value() % TAU, np.sin(theta.get_value() % TAU)))
        )
        self.add(graph_dot)

        self.play(theta.animate.set_value(TAU), run_time=4, rate_func=linear)
        
        star1 = Star(n=5, color=line4_color, fill_opacity=1).scale(0.1)
        star2 = Star(n=5, color=line4_color, fill_opacity=1).scale(0.1)
        
        star1.move_to(axes.c2p(PI, 0))
        star2.move_to(axes.c2p(0, 0))
        
        self.play(FadeIn(star1), FadeIn(star2))
        
        globe_star1 = Star(n=5, color=line4_color, fill_opacity=1).scale(0.1)
        globe_star2 = Star(n=5, color=line4_color, fill_opacity=1).scale(0.1)
        
        globe_star1.move_to(center + np.array([1.2 * np.cos(PI), 0.3 * np.sin(PI), 0]))
        globe_star2.move_to(center + np.array([1.2 * np.cos(0), 0.3 * np.sin(0), 0]))
        
        self.play(FadeIn(globe_star1), FadeIn(globe_star2))

        # === Animation for Lecture Line 5 ===
        line5_color = "#FFFFFF"
        self.play(self.lecture[4].animate.set_color(line5_color))
        
        ivt_text = Text("Intermediate Value Theorem", font_size=20, color=line5_color)
        # Fix for Issue 28: Expand placement area to prevent cramping and cutoff
        self.place_in_area(ivt_text, "F2", "F5", scale_factor=0.8)
        self.play(Write(ivt_text))
        
        self.play(theta.animate.set_value(TAU + PI/4), run_time=2)
        self.wait(2)
