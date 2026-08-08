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
        lecture_lines = [
            "Multiplying a vector by a number scales its length.",
            "Positive numbers stretch or shrink the arrow.",
            "Negative numbers flip the vector's direction entirely.",
            "The line's slope remains exactly the same.",
            "Scalar multiplication changes magnitude without changing the span."
        ]
        self.setup_layout("Scalar Multiplication: Growing and Shrinking", lecture_lines)
        
        # Grid setup for visual area - Using A3 to F6 to avoid overlap with lecture text (Issue 24)
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-5, 5, 1],
            x_length=4.5,
            y_length=5.0,
            background_line_style={"stroke_opacity": 0.2}
        )
        self.place_in_area(plane, 'A3', 'F6', scale_factor=0.85)
        origin = plane.c2p(0, 0)
        
        def get_p(x, y):
            return plane.c2p(x, y)

        # === Animation for Lecture Line 1 ===
        # Show a white (#FFFFFF) vector V at coordinates [1, 2] on the grid.
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        v_white = "#FFFFFF"
        vec = Arrow(start=origin, end=get_p(1, 2), buff=0, color=v_white)
        label = MathTex(r"\vec{v}", color=v_white, font_size=24)
        # Using buff=0.5 to prevent overcrowding and ensure clarity (Issue 25)
        label.next_to(vec.get_end(), UR, buff=0.5)
        
        self.play(Create(plane), GrowArrow(vec), Write(label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Positive numbers stretch or shrink the arrow.
        # Animate vector V stretching to twice its length [2, 4] while changing color to green (#00FF00).
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FF00")
        )
        
        v_green = "#00FF00"
        vec_2 = Arrow(start=origin, end=get_p(2, 4), buff=0, color=v_green)
        label_2 = MathTex(r"2\vec{v}", color=v_green, font_size=24)
        label_2.next_to(vec_2.get_end(), UR, buff=0.5)
        
        self.play(
            ReplacementTransform(vec, vec_2),
            ReplacementTransform(label, label_2)
        )
        self.wait(1)

        # Scale the green vector down to half its original length [0.5, 1] while changing color to yellow (#FFFF00).
        # We transition color within the same lecture line 2 logic.
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        v_yellow = "#FFFF00"
        vec_3 = Arrow(start=origin, end=get_p(0.5, 1), buff=0, color=v_yellow)
        label_3 = MathTex(r"0.5\vec{v}", color=v_yellow, font_size=24)
        # Note: buff=0.5 ensures labels don't crowd the origin even when vectors are small
        label_3.next_to(vec_3.get_end(), UR, buff=0.5)
        
        self.play(
            ReplacementTransform(vec_2, vec_3),
            ReplacementTransform(label_2, label_3)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Negative numbers flip the vector's direction entirely.
        # Flip the yellow vector to point in the opposite direction [-1, -2] and change color to red (#FF0000).
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FF0000")
        )
        
        v_red = "#FF0000"
        vec_4 = Arrow(start=origin, end=get_p(-1, -2), buff=0, color=v_red)
        label_4 = MathTex(r"-\vec{v}", color=v_red, font_size=24)
        label_4.next_to(vec_4.get_end(), DL, buff=0.5)
        
        self.play(
            ReplacementTransform(vec_3, vec_4),
            ReplacementTransform(label_3, label_4)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The line's slope remains exactly the same.
        # Show a thin dashed line (#333333) extending through the origin to show the span of the vector.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#888888")
        )
        
        # Calculate endpoints for the span line based on the plane's visible area
        span_line = DashedLine(
            start=get_p(-2.2, -4.4),
            end=get_p(2.2, 4.4),
            color="#333333",
            stroke_width=2
        )
        
        self.play(Create(span_line))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Scalar multiplication changes magnitude without changing the span.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(WHITE)
        )
        self.wait(2)
