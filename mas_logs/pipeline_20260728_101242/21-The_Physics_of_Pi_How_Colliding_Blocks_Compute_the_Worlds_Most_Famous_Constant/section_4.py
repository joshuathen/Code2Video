from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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
        lecture_lines = [
            "The energy ellipse makes the geometry difficult to solve.",
            "Let's rescale the large block's velocity axis.",
            "This transformation stretches the ellipse into a circle.",
            "Now, every collision is a point on this circle.",
            "The system's state moves along a circular arc."
        ]
        self.setup_layout("The Phase Space Transformation", lecture_lines)

        # Colors
        ELLIPSE_COLOR = WHITE
        FORMULA_COLOR = "#ADFF2F"
        CIRCLE_COLOR = WHITE
        POINT_COLOR = RED
        ARC_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(ELLIPSE_COLOR)
        
        # Create axes for phase space
        # x: v_2 (Large block), y: v_1 (Small block)
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-3, 3, 1],
            x_length=3,
            y_length=4,
            axis_config={"include_tip": True, "color": GREY_C}
        )
        axes_labels = axes.get_axis_labels(x_label="v_2", y_label="v_1")
        
        # A tall thin ellipse representing the energy state before transformation
        ellipse = Ellipse(width=1.5, height=3.5, color=ELLIPSE_COLOR)
        
        phase_space_group = VGroup(axes, axes_labels, ellipse)
        self.place_in_area(phase_space_group, "B2", "F6", scale_factor=0.8)
        
        self.play(Create(axes), Write(axes_labels))
        self.play(Create(ellipse))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(FORMULA_COLOR)
        
        # Display the scaling transformation v2 -> sqrt(M/m)v2
        # Resolving Issue 29: Position formula across A2-A5 for better centering
        transformation_formula = MathTex(
            r"v_2' = \sqrt{\frac{M}{m}} v_2",
            color=FORMULA_COLOR,
            font_size=36
        )
        self.place_in_area(transformation_formula, 'A2', 'A5', scale_factor=1.0)
        
        self.play(Write(transformation_formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(CIRCLE_COLOR)
        
        # Transform ellipse into a circle (horizontal stretch)
        circle = Circle(radius=1.75, color=CIRCLE_COLOR)
        circle.move_to(ellipse.get_center())
        
        # Update axes to match the new coordinate system
        new_axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": GREY_C}
        ).move_to(axes.get_center())
        
        # Create new label for rescaled axis
        new_x_label = MathTex("v_2'", font_size=24, color=FORMULA_COLOR)
        new_x_label.move_to(axes_labels[0].get_center())
        
        self.play(
            ReplacementTransform(ellipse, circle),
            ReplacementTransform(axes, new_axes),
            ReplacementTransform(axes_labels[0], new_x_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(WHITE)
        
        # Show points (dots) on the circle representing collision states
        angles = [170, 150, 130, 110, 90]
        collision_dots = VGroup(*[
            Dot(circle.point_at_angle(a * DEGREES), color=POINT_COLOR, radius=0.08) 
            for a in angles
        ])
        
        self.play(LaggedStart(*[FadeIn(dot) for dot in collision_dots], lag_ratio=0.2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(WHITE)
        
        # Trace a point moving along an arc on the circle
        start_angle = 90
        end_angle = 30
        
        arc_trace = Arc(
            radius=1.75, 
            start_angle=start_angle * DEGREES, 
            angle=(end_angle - start_angle) * DEGREES, 
            color=ARC_COLOR
        ).move_to(circle.get_center())
        
        moving_dot = Dot(color=YELLOW, radius=0.1)
        moving_dot.move_to(circle.point_at_angle(start_angle * DEGREES))
        
        # Use ValueTracker for the moving point
        angle_tracker = ValueTracker(start_angle)
        moving_dot.add_updater(lambda m: m.move_to(circle.point_at_angle(angle_tracker.get_value() * DEGREES)))
        
        self.add(moving_dot)
        self.play(
            Create(arc_trace),
            angle_tracker.animate.set_value(end_angle),
            run_time=3,
            rate_func=smooth
        )
        moving_dot.clear_updaters()
        self.wait(2)
