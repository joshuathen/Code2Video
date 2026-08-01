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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup the layout with section-specific title and lines
        # Updated script lines per Issue 45
        self.setup_layout(
            "The Special Case: Reaching π", 
            [
                'Rotate exactly halfway around, to an angle pi.', 
                'We land precisely at negative one.', 
                'Thus, e to the i pi equals negative one.'
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Color line 1 cyan to represent the angle setup
        self.play(self.lecture[0].animate.set_color("#00FFFF"))

        # Create Complex Plane (Axes)
        # Using a fixed area for the plane
        plane_axes = Axes(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=3.5,
            y_length=3.5,
            axis_config={"color": BLUE_B, "include_tip": True}
        )
        self.place_in_area(plane_axes, 'B2', 'E5')

        # Axis labels - using Text to avoid LaTeX dependency
        # Positions updated per Issue 38, 39, and 45
        label_real = Text("Re", font_size=16, color=BLUE_B)
        label_imag = Text("Im", font_size=16, color=BLUE_B)
        self.place_in_area(label_real, 'C6', 'D6', scale_factor=0.8)
        self.place_in_area(label_imag, 'A3', 'A4', scale_factor=0.8)

        # Unit Circle
        unit_circle = Circle(radius=plane_axes.get_x_unit_size(), color=WHITE, stroke_opacity=0.3)
        unit_circle.move_to(plane_axes.c2p(0, 0))

        # Initial point at (1, 0)
        start_dot = Dot(plane_axes.c2p(1, 0), color="#00FFFF")
        # Replacing MathTex with Text; positioning label near the point using grid
        label_one = Text("1", font_size=20, color=WHITE)
        self.place_at_grid(label_one, 'D5', scale_factor=0.8).shift(RIGHT*0.2 + DOWN*0.2)

        self.play(Create(plane_axes), Create(unit_circle))
        self.play(FadeIn(start_dot), Write(label_one), Write(label_real), Write(label_imag))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color line 2 gold to match the arc path
        self.play(self.lecture[1].animate.set_color("#FFD700"))

        # Value tracker for the angle from 0 to PI
        angle_tracker = ValueTracker(0)

        # The traveling dot
        moving_dot = Dot(color="#FFD700")
        moving_dot.add_updater(lambda d: d.move_to(
            plane_axes.c2p(np.cos(angle_tracker.get_value()), np.sin(angle_tracker.get_value()))
        ))
        
        # The gold arc tracing the path
        # Using a persistent arc that is updated/redrawn to avoid heavyweight recreation
        tracing_arc = always_redraw(lambda: Arc(
            radius=plane_axes.get_x_unit_size(),
            start_angle=0,
            angle=angle_tracker.get_value(),
            arc_center=plane_axes.c2p(0, 0),
            color="#FFD700",
            stroke_width=4
        ))

        self.add(tracing_arc, moving_dot)
        self.play(FadeOut(start_dot), FadeOut(label_one))
        
        # Animate the angle to PI
        self.play(angle_tracker.animate.set_value(PI), run_time=3, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color line 3 white to represent the final identity
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))

        # Highlight negative one - using Text for label
        label_neg_one = Text("-1", font_size=20, color=WHITE)
        self.place_at_grid(label_neg_one, 'D2', scale_factor=0.8).shift(LEFT*0.2 + DOWN*0.2)
        
        target_marker = Cross(scale_factor=0.1, stroke_width=2, color=RED).move_to(plane_axes.c2p(-1, 0))
        
        self.play(Create(target_marker), Write(label_neg_one))
        self.play(Indicate(label_neg_one, color="#FFFFFF"))

        # Resulting Formula - using MarkupText to avoid LaTeX dependency
        formula = MarkupText("e<sup>iπ</sup> = -1", font_size=36, color=WHITE)
        # Position updated per Issue 40 and 45
        self.place_in_area(formula, 'F2', 'F5', scale_factor=1.0)

        # Replace the dot with the formula
        self.play(
            FadeOut(moving_dot),
            FadeIn(formula),
            target_marker.animate.set_stroke(opacity=0)
        )
        self.wait(2)
