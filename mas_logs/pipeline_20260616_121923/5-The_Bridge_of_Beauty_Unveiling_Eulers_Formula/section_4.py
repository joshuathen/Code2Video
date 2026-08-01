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
        lecture_lines = [
            'Euler’s formula transforms growth into circular rotation.', 
            'As x increases, we move along the unit circle.', 
            "The point travels like a clock's steady hand.", 
            'It never flies away, but orbits the center.', 
            'Imaginary growth pulls the path into a perfect ring.'
        ]
        self.setup_layout("The Transformation: From Growth to Rotation", lecture_lines)

        # Pre-calculate components
        # 1. Complex Plane and Circle
        plane = ComplexPlane(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=4,
            y_length=4,
            axis_config={"color": GREY}
        )
        # Using Text for axis labels to avoid LaTeX dependency issues
        re_label = Text("Re", font_size=18).next_to(plane.x_axis.get_end(), RIGHT, buff=0.1)
        im_label = Text("Im", font_size=18).next_to(plane.y_axis.get_top(), UP, buff=0.1)
        
        circle = Circle(radius=plane.get_x_unit_size(), color="#808080")
        circle.move_to(plane.get_origin())
        
        # Group for positioning (Issue 41)
        complex_plane_group = VGroup(plane, re_label, im_label, circle)
        self.place_in_area(complex_plane_group, 'A2', 'E5', scale_factor=0.8)

        # 2. Formula (Issue 42)
        euler_formula = Text("e^ix = cos(x) + i sin(x)", font_size=32, color=WHITE)
        self.place_in_area(euler_formula, 'F2', 'F6', scale_factor=0.7)

        # 3. Dynamic Tracking Elements
        x_tracker = ValueTracker(0)

        # Tracing Point
        tracing_point = Dot(color=BLUE)
        tracing_point.add_updater(lambda d: d.move_to(
            plane.coords_to_point(np.cos(x_tracker.get_value()), np.sin(x_tracker.get_value()))
        ))

        # Projection Lines (Dashed Yellow #FFFF00)
        cos_projection = DashedLine(color="#FFFF00", stroke_width=2)
        cos_projection.add_updater(lambda l: l.put_start_and_end_on(
            plane.coords_to_point(np.cos(x_tracker.get_value()), 0),
            plane.coords_to_point(np.cos(x_tracker.get_value()), np.sin(x_tracker.get_value()))
        ))

        sin_projection = DashedLine(color="#FFFF00", stroke_width=2)
        sin_projection.add_updater(lambda l: l.put_start_and_end_on(
            plane.coords_to_point(0, np.sin(x_tracker.get_value())),
            plane.coords_to_point(np.cos(x_tracker.get_value()), np.sin(x_tracker.get_value()))
        ))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(plane), Write(re_label), Write(im_label))
        self.play(Create(circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Write(euler_formula))
        self.play(FadeIn(tracing_point), Create(cos_projection), Create(sin_projection))
        # Start rotation
        self.play(x_tracker.animate.set_value(PI/2), run_time=2, rate_func=linear)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        # Continue rotation like a clock hand
        self.play(x_tracker.animate.set_value(PI), run_time=2, rate_func=linear)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Highlight specific point (Issue 43)
        identity_label = Text("e^iπ = -1", font_size=28, color=RED)
        self.place_at_grid(identity_label, 'C2', scale_factor=0.6)
        
        self.play(Write(identity_label))
        # Pulsing effect
        self.play(tracing_point.animate.scale(1.5).set_color(RED), run_time=0.4)
        self.play(tracing_point.animate.scale(1/1.5).set_color(BLUE), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        # Complete the orbit
        self.play(x_tracker.animate.set_value(2*PI), run_time=3, rate_func=linear)
        self.wait(2)

        # Cleanup
        self.play(
            FadeOut(complex_plane_group),
            FadeOut(euler_formula),
            FadeOut(tracing_point),
            FadeOut(cos_projection),
            FadeOut(sin_projection),
            FadeOut(identity_label),
            FadeOut(self.title),
            FadeOut(self.lecture)
        )
