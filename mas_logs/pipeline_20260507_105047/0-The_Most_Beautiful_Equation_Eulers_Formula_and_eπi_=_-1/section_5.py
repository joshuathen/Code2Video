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
        # 1. Configuration and Layout
        title_text = "The Journey of π: Reaching the Destination"
        lecture_lines = [
            'Set the rotation angle to exactly pi radians.',
            'Pi radians corresponds to a perfect half-circle.',
            'We start at positive one on the real axis.',
            'A half-circle rotation lands us at negative one.',
            'Therefore, e to the power i pi is negative one.'
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors for lecture lines
        colors = [BLUE_B, GREEN_B, YELLOW_B, RED_B, PURPLE_B]

        # === Animation for Lecture Line 1 ===
        # Set the rotation angle to exactly pi radians.
        self.play(self.lecture[0].animate.set_color(colors[0]))
        
        plane = ComplexPlane(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            background_line_style={"stroke_opacity": 0.4}
        )
        # Issue 42 Fix: complex plane and circle animation lack a defined grid area
        self.place_in_area(plane, 'B1', 'F6', scale_factor=0.8)
        
        circle = Circle(radius=plane.get_x_unit_size(), color=WHITE, stroke_width=2)
        circle.move_to(plane.n2p(0))
        
        self.play(Create(plane), Create(circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Pi radians corresponds to a perfect half-circle.
        self.play(self.lecture[1].animate.set_color(colors[1]))
        
        # Draw arc representing pi radians
        arc = Arc(
            radius=plane.get_x_unit_size(), 
            start_angle=0, 
            angle=PI, 
            color=colors[1],
            arc_center=plane.n2p(0)
        )
        self.play(Create(arc))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We start at positive one on the real axis.
        self.play(self.lecture[2].animate.set_color(colors[2]))
        
        # Starting point (1,0)
        vector = Arrow(plane.n2p(0), plane.n2p(1), buff=0, color=colors[2])
        dot = Dot(plane.n2p(1), color=colors[2])
        label_one = MathTex("1", color=WHITE)
        # Issue 43 Fix 1: Labels for start point anchored to grid
        self.place_at_grid(label_one, 'C5', scale_factor=0.7)
        
        self.play(GrowArrow(vector), FadeIn(dot), Write(label_one))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # A half-circle rotation lands us at negative one.
        self.play(self.lecture[3].animate.set_color(colors[3]))
        
        label_minus_one = MathTex("-1", color=WHITE)
        # Issue 43 Fix 2: Labels for end point anchored to grid
        self.place_at_grid(label_minus_one, 'C1', scale_factor=0.7)
        
        # Animate rotation of the vector using ValueTracker for persistence
        angle_tracker = ValueTracker(0)
        vector.add_updater(
            lambda v: v.become(
                Arrow(
                    plane.n2p(0), 
                    plane.n2p(np.exp(1j * angle_tracker.get_value())), 
                    buff=0, 
                    color=colors[3]
                )
            )
        )
        dot.add_updater(
            lambda d: d.move_to(plane.n2p(np.exp(1j * angle_tracker.get_value())))
        )
        
        self.play(angle_tracker.animate.set_value(PI), run_time=2, rate_func=linear)
        vector.clear_updaters()
        dot.clear_updaters()
        
        self.play(Write(label_minus_one))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Therefore, e to the power i pi is negative one.
        self.play(self.lecture[4].animate.set_color(colors[4]))
        
        euler_formula = MathTex("e^{i\pi} = -1", color=WHITE)
        # Issue 44 Fix: Position Euler's identity formula
        self.place_in_area(euler_formula, 'A2', 'A5', scale_factor=1.0)
        
        self.play(Write(euler_formula))
        self.wait(1)
        
        # Transform the identity into e^{i\pi} + 1 = 0
        final_formula = MathTex("e^{i\pi} + 1 = 0", color=WHITE)
        self.place_in_area(final_formula, 'A2', 'A5', scale_factor=1.0)
        
        self.play(Transform(euler_formula, final_formula))
        self.wait(1)
        
        # Draw a glowing gold box around the final formula
        box = SurroundingRectangle(euler_formula, color="#FFD700", buff=0.2)
        self.play(Create(box), euler_formula.animate.set_color("#FFD700"))
        self.wait(2)
