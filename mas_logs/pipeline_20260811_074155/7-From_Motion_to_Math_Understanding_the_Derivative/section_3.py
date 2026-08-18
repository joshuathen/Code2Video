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
        # Setup the layout
        self.setup_layout("The Concept of 'h': Zooming In", 
                          ["Let's call the distance between our two points 'h'.", 
                           "To find instantaneous speed, we must shrink 'h'.", 
                           "Watch the gap narrow as points come together."])
        
        # Colors
        h_color = "#ADD8E6" # Light Blue
        curve_color = YELLOW
        point_color = WHITE

        # 1. Setup Axes and Curve
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 5, 1],
            axis_config={"include_tip": True, "color": WHITE},
            x_length=4,
            y_length=4
        )
        # Resolved Issues 28 and 29: adjust placement area and scale to prevent bottom-cluttering and excessive whitespace
        self.place_in_area(axes, 'B2', 'E5', scale_factor=0.9)
        
        def func(x):
            # A simple parabola-like curve
            return 0.3 * x**2 + 0.5
        
        graph = axes.plot(func, x_range=[0, 3.5], color=curve_color)
        
        # 2. Points and Trackers
        h_tracker = ValueTracker(1.5)
        x_a = 1.0
        
        # Point A (Fixed relative to axes)
        dot_a = Dot(axes.c2p(x_a, func(x_a)), color=point_color)
        label_a = MathTex("A", font_size=20).next_to(dot_a, UP + LEFT, buff=0.1)
        
        # Point B (Dynamic)
        dot_b = Dot(axes.c2p(x_a + h_tracker.get_value(), func(x_a + h_tracker.get_value())), color=point_color)
        dot_b.add_updater(lambda d: d.move_to(axes.c2p(x_a + h_tracker.get_value(), func(x_a + h_tracker.get_value()))))
        
        label_b = MathTex("B", font_size=20)
        label_b.add_updater(lambda l: l.next_to(dot_b, UP + RIGHT, buff=0.1))
        
        # Vertical indicators
        line_a = axes.get_vertical_line(dot_a.get_center(), color=GRAY, stroke_width=2)
        # Using a Line for line_b to allow dynamic updating
        line_b = Line(axes.c2p(x_a + h_tracker.get_value(), func(x_a + h_tracker.get_value())), axes.c2p(x_a + h_tracker.get_value(), 0), color=GRAY, stroke_width=2)
        line_b.add_updater(lambda l: l.put_start_and_end_on(axes.c2p(x_a + h_tracker.get_value(), func(x_a + h_tracker.get_value())), axes.c2p(x_a + h_tracker.get_value(), 0)))

        # Labels on X-axis
        label_xa = MathTex("x", font_size=20).next_to(axes.c2p(x_a, 0), DOWN, buff=0.1)
        label_xb = MathTex("x+h", font_size=20, color=h_color)
        label_xb.add_updater(lambda l: l.next_to(axes.c2p(x_a + h_tracker.get_value(), 0), DOWN, buff=0.1))

        # Interval 'h' representation (using a simple line for performance)
        h_line = Line(axes.c2p(x_a, -0.4), axes.c2p(x_a + h_tracker.get_value(), -0.4), color=h_color)
        h_line.add_updater(lambda l: l.put_start_and_end_on(axes.c2p(x_a, -0.4), axes.c2p(x_a + h_tracker.get_value(), -0.4)))
        
        h_label = MathTex("h", font_size=24, color=h_color)
        h_label.add_updater(lambda l: l.next_to(h_line, DOWN, buff=0.05))

        # Group everything for the final zoom
        main_visuals = VGroup(axes, graph, dot_a, label_a, dot_b, label_b, line_a, line_b, label_xa, label_xb, h_line, h_label)

        # === Animation for Lecture Line 1 ===
        # "Let's call the distance between our two points 'h'."
        self.lecture[0].set_color(h_color)
        self.play(
            Create(axes),
            Create(graph),
            FadeIn(dot_a),
            Write(label_a),
            FadeIn(dot_b),
            Write(label_b),
            Create(line_a),
            Create(line_b),
            Write(label_xa),
            Write(label_xb),
            Create(h_line),
            Write(h_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "To find instantaneous speed, we must shrink 'h'."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(h_color)
        
        self.play(
            h_tracker.animate.set_value(0.6),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Watch the gap narrow as points come together."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(h_color)
        
        # Zoom target is point A
        zoom_point = dot_a.get_center()
        
        self.play(
            h_tracker.animate.set_value(0.15),
            main_visuals.animate.scale(2.5, about_point=zoom_point),
            run_time=4
        )
        self.wait(2)
        
        # Reset color at the end
        self.lecture[2].set_color(WHITE)
