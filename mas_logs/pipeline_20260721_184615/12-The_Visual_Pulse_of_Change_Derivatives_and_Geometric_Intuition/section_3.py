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
        self.setup_layout(
            "Geometric Intuition: The 'Zoom-In' Effect",
            [
                "Watch as point B slides closer to point A.",
                "The horizontal distance between them shrinks toward zero.",
                "The secant line transforms into a touching tangent line.",
                "Zooming in makes the curve look perfectly straight.",
                "This tangent's slope is the derivative at that point."
            ]
        )

        # Setup Graph
        # Issue 26 Fix: Place axes at D5 to prevent expansion into lecture text
        axes = Axes(
            x_range=[-0.5, 2.5, 1],
            y_range=[-0.5, 3.5, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"include_tip": False, "color": "#FFFFFF"}
        )
        self.place_at_grid(axes, "D5")

        # Function f(x) = x^2 / 2
        def func(x):
            return 0.5 * x**2

        curve = axes.plot(func, x_range=[0, 2.2], color="#FFFFFF")
        
        # Point A (fixed)
        x_a = 0.8
        point_a_coords = axes.c2p(x_a, func(x_a))
        dot_a = Dot(point_a_coords, color="#FFFFFF", radius=0.06)
        label_a = MathTex("A", font_size=24, color="#FFFFFF").next_to(dot_a, LEFT, buff=0.1)

        # ValueTracker for Delta X
        dx_tracker = ValueTracker(1.2)
        
        # Dynamic Dot B
        dot_b = Dot(color="#FFFFFF", radius=0.06)
        dot_b.add_updater(lambda d: d.move_to(axes.c2p(x_a + dx_tracker.get_value(), func(x_a + dx_tracker.get_value()))))
        
        label_b = MathTex("B", font_size=24, color="#FFFFFF")
        label_b.add_updater(lambda l: l.next_to(dot_b, RIGHT, buff=0.1))

        # Secant Line using persistent mobject pattern
        secant_line = Line(color="#FFFF00")
        def update_secant(l):
            p1 = axes.c2p(x_a, func(x_a))
            p2 = axes.c2p(x_a + dx_tracker.get_value(), func(x_a + dx_tracker.get_value()))
            # Create a line segment between A and B
            # Use small epsilon to prevent zero-length line errors
            dist = np.linalg.norm(p2 - p1)
            if dist < 0.001:
                return
            new_line = Line(p1, p2)
            # Scale it to make it look like an extended secant line
            new_line.scale(4, about_point=p1)
            l.set_points(new_line.get_points())

        secant_line.add_updater(update_secant)

        # Tangent Line (for morphing later)
        # f'(x) = x. At x=0.8, slope is 0.8.
        tangent_line_const = Line(
            axes.c2p(x_a - 1.2, func(x_a) - 0.8 * 1.2),
            axes.c2p(x_a + 1.2, func(x_a) + 0.8 * 1.2),
            color="#FF0000"
        )

        # Group for visual components
        visual_group = VGroup(axes, curve, dot_a, label_a)

        # === Animation for Lecture Line 1 ===
        # Watch as point B slides closer to point A.
        self.lecture[0].set_color("#FFFFFF")
        self.add(visual_group, dot_b, label_b, secant_line)
        self.play(dx_tracker.animate.set_value(0.6), run_time=3, rate_func=rate_functions.smooth)
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # The horizontal distance between them shrinks toward zero.
        self.lecture[0].set_color(GRAY)
        self.lecture[1].set_color("#FF69B4")
        
        # Issue 27 Fix: Position dx_label at C5 with scale 0.8 for proximity
        dx_label = MathTex(r"\Delta x \to 0", color="#FF69B4", font_size=32)
        self.place_at_grid(dx_label, "C5", scale_factor=0.8)
        
        self.play(Write(dx_label))
        self.play(Indicate(dx_label, color="#FF69B4", scale_factor=1.1))
        self.play(dx_tracker.animate.set_value(0.1), run_time=3)
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # The secant line transforms into a touching tangent line.
        self.lecture[1].set_color(GRAY)
        self.lecture[2].set_color("#FFFF00")
        
        # Freeze secant line for morphing
        secant_at_morph = secant_line.copy().clear_updaters()
        self.remove(secant_line)
        self.add(secant_at_morph)
        
        self.play(dx_tracker.animate.set_value(0.01), run_time=2)
        self.play(ReplacementTransform(secant_at_morph, tangent_line_const))
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        # Zooming in makes the curve look perfectly straight.
        self.lecture[2].set_color(GRAY)
        self.lecture[3].set_color("#FFFFFF")
        
        # Remove point B and labels for zoom clarity
        self.remove(dot_b, label_b, dx_label)
        
        # Group everything for the zoom
        final_group = VGroup(visual_group, tangent_line_const)
        self.play(
            final_group.animate.scale(4.5, about_point=point_a_coords),
            run_time=3
        )
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # This tangent's slope is the derivative at that point.
        self.lecture[3].set_color(GRAY)
        self.lecture[4].set_color("#FF0000")
        
        # Issue 28 Fix: Place derivative label in area B5-B6 to avoid overlap with label A
        derivative_label = Text("Derivative = Slope", font_size=24, color="#FF0000")
        self.place_in_area(derivative_label, "B5", "B6", scale_factor=0.7)
        
        self.play(Write(derivative_label))
        self.play(Indicate(derivative_label, color="#FF0000"))
        self.wait(2.0)
