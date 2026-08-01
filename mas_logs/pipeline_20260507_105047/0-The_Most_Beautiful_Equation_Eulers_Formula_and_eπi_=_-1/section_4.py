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
        # Setup title and lecture lines
        lecture_lines = [
            "Real exponents scale a value up or down.",
            "Imaginary exponents do something different: they rotate.",
            "The expression e to the i theta traces a circle.",
            "Here, theta represents the angle of rotation.",
            "Growth is transformed into continuous circular motion."
        ]
        self.setup_layout("The Secret of Imaginary Exponents", lecture_lines)
        
        # Visual color palette
        COLOR_THETA = YELLOW
        COLOR_COS = BLUE
        COLOR_SIN = RED
        COLOR_CIRCLE = "#555555"
        COLOR_FORMULA = WHITE

        # === Animation for Lecture Line 1 ===
        # Real exponents scale a value up or down.
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Imaginary exponents do something different: they rotate.
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        # Create Complex Plane
        plane = NumberPlane(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_color": BLUE_D, "stroke_width": 1, "stroke_opacity": 0.3}
        )
        # Visual Anchor System: Positioning plane to avoid lecture text
        plane_group = VGroup(plane)
        self.place_in_area(plane_group, 'B1', 'F6', scale_factor=0.8) # Issue 40 resolution
        
        self.play(Create(plane))
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # The expression e to the i theta traces a circle.
        self.play(self.lecture[2].animate.set_color(WHITE))
        
        # Euler's formula display
        euler_formula = Text("e^iθ = cos(θ) + i sin(θ)", font_size=32, color=COLOR_FORMULA)
        self.place_in_area(euler_formula, 'A1', 'A6', scale_factor=0.7) # Issue 41 resolution
        
        # Grey unit circle (#555555)
        radius_val = plane.get_x_unit_size()
        unit_circle = Circle(radius=radius_val, color=COLOR_CIRCLE)
        unit_circle.move_to(plane.coords_to_point(0, 0))
        
        self.play(Write(euler_formula), Create(unit_circle))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Here, theta represents the angle of rotation.
        self.play(self.lecture[3].animate.set_color(COLOR_THETA))
        
        # Angle tracker and rotating vector
        theta_tracker = ValueTracker(0.01)
        
        # Rotating Vector
        vector = Arrow(
            start=plane.coords_to_point(0,0), 
            end=plane.coords_to_point(1,0), 
            buff=0, 
            color=COLOR_THETA,
            max_tip_length_to_length_ratio=0.15
        )
        vector.add_updater(lambda v: v.put_start_and_end_on(
            plane.coords_to_point(0,0),
            plane.coords_to_point(np.cos(theta_tracker.get_value()), np.sin(theta_tracker.get_value()))
        ))
        
        # Angle arc and label
        # We avoid heavyweight redraws for text
        arc = always_redraw(lambda: Arc(
            radius=0.4 * radius_val,
            start_angle=0,
            angle=theta_tracker.get_value(),
            color=COLOR_THETA,
            arc_center=plane.coords_to_point(0,0)
        ))
        
        theta_label = Text("θ", color=COLOR_THETA, font_size=24)
        theta_label.add_updater(lambda m: m.move_to(
            plane.coords_to_point(
                0.6 * np.cos(theta_tracker.get_value()/2), 
                0.6 * np.sin(theta_tracker.get_value()/2)
            )
        ))
        
        self.play(GrowArrow(vector), Create(arc), Write(theta_label))
        
        # Initial rotation to show the concept
        self.play(theta_tracker.animate.set_value(PI/3), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # Growth is transformed into continuous circular motion.
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        # Projections to axes: cos(theta) and i sin(theta)
        cos_line = Line(color=COLOR_COS, stroke_width=6)
        cos_line.add_updater(lambda l: l.set_points_as_corners([
            plane.coords_to_point(0,0),
            plane.coords_to_point(np.cos(theta_tracker.get_value()), 0)
        ]))
        
        sin_line = Line(color=COLOR_SIN, stroke_width=6)
        sin_line.add_updater(lambda l: l.set_points_as_corners([
            plane.coords_to_point(0,0),
            plane.coords_to_point(0, np.sin(theta_tracker.get_value()))
        ]))
        
        cos_label = Text("cos(θ)", color=COLOR_COS, font_size=20)
        cos_label.add_updater(lambda m: m.next_to(cos_line, DOWN, buff=0.1))
        
        sin_label = Text("i sin(θ)", color=COLOR_SIN, font_size=20)
        sin_label.add_updater(lambda m: m.next_to(sin_line, LEFT, buff=0.1))
        
        self.play(
            Create(cos_line), 
            Create(sin_line), 
            FadeIn(cos_label), 
            FadeIn(sin_label)
        )
        
        # Demonstrate full rotation showing continuous circular motion
        self.play(theta_tracker.animate.set_value(2*PI), run_time=6, rate_func=linear)
        self.wait(2)
