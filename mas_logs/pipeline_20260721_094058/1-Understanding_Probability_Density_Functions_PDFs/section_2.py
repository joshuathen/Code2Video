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
        lecture_lines = [
            "A single point on the axis has no width.",
            "An exact value like five kilograms is nearly impossible.",
            "Thus, the probability at any single point is zero.",
            "We must define probabilities over a range of values.",
            "Shaded intervals represent the probability of an event occurring."
        ]
        self.setup_layout("The Zero Probability Paradox", lecture_lines)

        # Colors
        RED_COLOR = "#FC6255"
        GREEN_COLOR = "#83C167"

        # Setup Main Axes and Curve
        axes = Axes(
            x_range=[3, 7, 1],
            y_range=[0, 1.2, 0.5],
            axis_config={"include_tip": False},
            x_length=5,
            y_length=4
        ).set_color(WHITE)
        curve = axes.plot(lambda x: np.exp(-(x - 5)**2 / 0.5), color=WHITE)
        graph_group = VGroup(axes, curve)
        self.place_in_area(graph_group, "B2", "F6", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(RED_COLOR)
        # Show a smooth curve with a thin red vertical line at x=5.
        point_line = axes.get_vertical_line(axes.c2p(5, 1.0), color=RED_COLOR)
        
        self.play(Create(axes), Create(curve))
        self.play(Create(point_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RED_COLOR)
        # Display the text '5.000... kg' with a red 'X' next to it.
        exact_value_text = Text("5.000... kg", font_size=20, color=RED_COLOR)
        cross = Cross(stroke_width=4).scale(0.2).set_color(RED_COLOR)
        val_group = VGroup(exact_value_text, cross).arrange(RIGHT, buff=0.2)
        # Fix for Issue 23: use place_in_area and lower scale
        self.place_in_area(val_group, "A3", "A5", scale_factor=0.8)
        
        self.play(Write(exact_value_text))
        self.play(Create(cross))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED_COLOR)
        # Zoom in on the red line to show it has no width.
        
        # New Axes for zoom view
        zoom_axes = Axes(
            x_range=[4.995, 5.005, 0.002],
            y_range=[0, 1.2, 0.5],
            axis_config={"include_tip": False},
            x_length=5,
            y_length=4
        ).set_color(WHITE)
        zoom_curve = zoom_axes.plot(lambda x: np.exp(-(x - 5)**2 / 0.5), color=WHITE)
        zoom_group = VGroup(zoom_axes, zoom_curve)
        self.place_in_area(zoom_group, "B2", "F6", scale_factor=0.8)
        
        # Create line relative to the zoomed axes
        zoom_line = zoom_axes.get_vertical_line(zoom_axes.c2p(5, 1.0), color=RED_COLOR)
        
        width_label = Text("Width = 0", font_size=20, color=RED_COLOR)
        # Fix for Issue 22: move from D4 to E4 and reduce scale
        self.place_at_grid(width_label, "E4", scale_factor=0.8)

        self.play(
            FadeOut(graph_group),
            FadeOut(point_line),
            FadeIn(zoom_group),
            FadeIn(zoom_line)
        )
        self.play(Write(width_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN_COLOR)
        # We must define probabilities over a range of values.
        
        # Highlight interval [4.9, 5.1]
        interval_highlight = Line(
            axes.c2p(4.9, 0),
            axes.c2p(5.1, 0),
            color=GREEN_COLOR,
            stroke_width=8
        )
        
        self.play(
            FadeOut(zoom_group),
            FadeOut(zoom_line),
            FadeOut(width_label),
            FadeOut(val_group),
            FadeIn(graph_group),
            FadeIn(point_line)
        )
        self.play(Create(interval_highlight))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(GREEN_COLOR)
        # Shaded intervals represent the probability of an event occurring.
        
        # Fill area under curve between 4.9 and 5.1
        area = axes.get_area(curve, x_range=[4.9, 5.1], color=GREEN_COLOR, opacity=0.5)
        
        self.play(FadeIn(area))
        self.wait(2)
