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
        lecture_lines = [
            "The Cost Function measures Pixel's prediction error.",
            "It looks like a U-shaped valley on a graph.",
            "A higher point means a much larger mistake.",
            "We call this function Mean Squared Error, or MSE.",
            "Our mission is to reach the valley's bottom."
        ]
        self.setup_layout("The Cost Function: The Scorecard of Error", lecture_lines)

        # Pre-creating visual elements
        # Axes
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"color": "#FFFFFF", "include_tip": True}
        )
        labels = axes.get_axis_labels(
            x_label=Text("Weight", font_size=16, color="#FFFFFF"),
            y_label=Text("Cost (Error)", font_size=16, color="#FFFFFF")
        )
        graph_group = VGroup(axes, labels)
        # Fix for Issue 34: use larger area and scale down to avoid clipping
        self.place_in_area(graph_group, "B2", "F5", scale_factor=0.8)

        # Parabola: J(w) = (w-2)^2 + 0.5
        parabola = axes.plot(
            lambda x: (x-2)**2 + 0.5,
            x_range=[0.3, 3.7],
            color="#00CCFF"
        )

        # Dynamic Point
        vt = ValueTracker(0.6) # Starting weight value (High error)
        red_point = Dot(color="#FF0000").scale(1.2)
        red_point.add_updater(lambda d: d.move_to(axes.c2p(vt.get_value(), (vt.get_value()-2)**2 + 0.5)))

        # Target (Bottom) - Asset integration Issue 26
        star = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/star.svg")
        star.set_color("#00FF00")
        star.scale(0.15)
        star.move_to(axes.c2p(2, 0.5))

        # MSE Formula/Label - Fix for Issue 35: Move to A5, scale down
        mse_label = Text("MSE Function", font_size=18, color="#FFFFFF")
        self.place_at_grid(mse_label, "A5", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        # The Cost Function measures Pixel's prediction error.
        self.play(self.lecture[0].animate.set_color("#00CCFF"))
        self.play(Create(axes), Write(labels))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # It looks like a U-shaped valley on a graph.
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#00CCFF"))
        self.play(Create(parabola))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A higher point means a much larger mistake.
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FF0000"))
        self.play(FadeIn(red_point))
        # Highlight position high on the curve
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # We call this function Mean Squared Error, or MSE.
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color("#FFFFFF"))
        self.play(Write(mse_label))
        # Move the point to show weight changing
        self.play(vt.animate.set_value(3.4), run_time=1.5)
        self.play(vt.animate.set_value(1.2), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Our mission is to reach the valley's bottom.
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color("#00FF00"))
        self.play(FadeIn(star))
        # Move point to bottom
        self.play(vt.animate.set_value(2.0), run_time=2, rate_func=slow_into)
        self.wait(2)
