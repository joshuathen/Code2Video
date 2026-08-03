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
        # Setup layout
        title_text = "The Concept of the Limit (h \u2192 0)"
        lecture_lines = [
            "Let the distance between points be 'h'.",
            "Watch as we move the points closer together.",
            "As 'h' shrinks, the interval nearly disappears.",
            "We are finding the limit as 'h' approaches zero.",
            "The curve starts to look like a straight line."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_H = "#FFFF00"
        COLOR_CURVE = "#87CEEB"
        COLOR_POINT = "#FF6347"

        # Coordinate system setup
        ax = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 3, 1],
            axis_config={"include_tip": False},
            x_length=5,
            y_length=5
        )
        # Fix: Adjust scale and position of axes per Critic feedback to avoid overlap
        self.place_in_area(ax, "A1", "F6", scale_factor=0.6)

        def func(x):
            return 0.2 * x**2 + 0.5

        curve = ax.plot(func, x_range=[0, 3], color=COLOR_CURVE)
        
        # Point A is fixed at a specific spot on the curve
        ax_pos_a = [1.2, func(1.2)]
        dot_a = Dot(ax.c2p(*ax_pos_a), color=COLOR_POINT)
        
        # Fix: Place label_a at specific grid point with smaller scale per Critic
        label_a = MathTex("A", font_size=24)
        self.place_at_grid(label_a, 'D3', scale_factor=0.4)

        # Value tracker for h (distance between points)
        h_tracker = ValueTracker(1.2)

        # Persistent mobjects for dynamic elements
        dot_b = Dot(color=COLOR_POINT)
        h_line = Line(color=COLOR_H, stroke_width=4)
        
        # Fix: Place h_label at specific grid point with smaller scale per Critic
        h_label = MathTex("h", color=COLOR_H, font_size=24)
        self.place_at_grid(h_label, 'D4', scale_factor=0.4)
        
        secant = Line(color=WHITE, stroke_width=2)

        # Updaters for dynamic elements
        def update_dot_b(d):
            h = h_tracker.get_value()
            d.move_to(ax.c2p(ax_pos_a[0] + h, func(ax_pos_a[0] + h)))

        def update_h_line(l):
            h = h_tracker.get_value()
            l.set_points_as_corners([
                ax.c2p(ax_pos_a[0], ax_pos_a[1]),
                ax.c2p(ax_pos_a[0] + h, ax_pos_a[1])
            ])

        def update_h_label(m):
            # Updater keeps the label near the interval line as it shrinks
            m.next_to(h_line, DOWN, buff=0.1)

        def update_secant(l):
            p1 = ax.c2p(*ax_pos_a)
            h = h_tracker.get_value()
            p2 = ax.c2p(ax_pos_a[0] + h, func(ax_pos_a[0] + h))
            vec = p2 - p1
            if np.linalg.norm(vec) > 0.001:
                u = vec / np.linalg.norm(vec)
                # Fix: Shortened line length (from 1.5/3.5 to 0.5/2.0) to avoid obstructing lecture notes on the left
                l.set_points_as_corners([p1 - u * 0.5, p1 + u * 2.0])
            else:
                # Tangent fallback
                p2_alt = ax.c2p(ax_pos_a[0] + 0.01, func(ax_pos_a[0] + 0.01))
                u = (p2_alt - p1) / np.linalg.norm(p2_alt - p1)
                l.set_points_as_corners([p1 - u * 0.5, p1 + u * 2.0])

        dot_b.add_updater(update_dot_b)
        h_line.add_updater(update_h_line)
        h_label.add_updater(update_h_label)
        secant.add_updater(update_secant)

        # === Animation for Lecture Line 1 ===
        # "Let the distance between points be 'h'."
        self.lecture[0].set_color(COLOR_H)
        self.play(Create(ax), Create(curve))
        self.play(FadeIn(dot_a), Write(label_a))
        
        # Zoom in on A as per storyboard
        zoom_center = ax.c2p(*ax_pos_a)
        all_visuals = VGroup(ax, curve, dot_a, label_a, dot_b, h_line, h_label)
        
        self.add(dot_b, h_line, h_label)
        self.play(
            all_visuals.animate.scale(2.5, about_point=zoom_center),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Watch as we move the points closer together."
        self.lecture[1].set_color(COLOR_H)
        self.play(Create(secant))
        self.play(h_tracker.animate.set_value(0.6), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "As 'h' shrinks, the interval nearly disappears."
        self.lecture[2].set_color(COLOR_H)
        self.play(h_tracker.animate.set_value(0.2), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "We are finding the limit as 'h' approaches zero."
        self.lecture[3].set_color(COLOR_H)
        self.play(h_tracker.animate.set_value(0.02), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "The curve starts to look like a straight line."
        self.lecture[4].set_color(COLOR_H)
        all_visuals.add(secant)
        # Final zoom to emphasize the linearity
        self.play(
            all_visuals.animate.scale(2, about_point=zoom_center),
            secant.animate.set_stroke(width=4),
            run_time=3
        )
        self.wait(2)
