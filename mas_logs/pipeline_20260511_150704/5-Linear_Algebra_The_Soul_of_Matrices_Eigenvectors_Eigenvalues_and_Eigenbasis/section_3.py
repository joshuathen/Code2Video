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
        # Initial lecture lines setup
        lines = [
            'Eigenvectors are special vectors staying on their span.',
            'Their direction remains constant during the transformation.',
            'The eigenvalue measures the amount of scaling.',
            'It shows if the vector stretches or flips.',
            'Formally, A times v equals lambda times v.'
        ]
        self.setup_layout("Core Definitions: Eigenvectors and Eigenvalues", lines)

        # === Animation for Lecture Line 1 ===
        # Create coordinate system
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_tip": True}
        )
        # Issue 40: Adjust plane placement
        self.place_in_area(plane, 'B3', 'F6', scale_factor=0.6)
        
        # Create Vector v at (1,1) relative to plane
        vector_v = Arrow(plane.coords_to_point(0, 0), plane.coords_to_point(1, 1), color="#00FFFF", buff=0)
        # Issue 38: Adjust vector arrow placement
        self.place_in_area(vector_v, 'C3', 'E5', scale_factor=0.6)
        
        # Create Span
        span = DashedLine(plane.coords_to_point(-3, -3), plane.coords_to_point(3, 3), color=GRAY, stroke_opacity=0.5)
        # Align span with vector's new position from place_in_area
        span.move_to(vector_v.get_start())
        span.shift(vector_v.get_vector() * 0.5) # Center span on vector
        
        v_label = Text("v", font_size=24, color="#00FFFF")
        # Issue 39: Adjust label placement
        self.place_at_grid(v_label, 'C5', scale_factor=0.8)

        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.play(Create(plane), Create(span))
        self.play(GrowArrow(vector_v), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        # Transform the plane and move vector to (2,2) equivalent
        # Since we decoupled from plane, we manually scale the vector to show it reaching (2,2)
        self.play(
            vector_v.animate.scale(2, about_point=vector_v.get_start()),
            v_label.animate.shift(UP * 0.5 + RIGHT * 0.5),
            plane.animate.apply_matrix([[2, 0], [0, 2]]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        # Overlay scaled vector lambda*v
        vector_lv = vector_v.copy().set_color("#FFFF00")
        self.play(FadeIn(vector_lv, scale=1.1))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#FFFF00")
        )
        # Show scaling visually by pulsing vector_lv
        self.play(vector_lv.animate.scale(1.1), rate_func=there_and_back)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(WHITE)
        )
        
        # Display Equation Av = lambda v
        # Using Text to avoid LaTeX dependency
        equation = Text("Av = λv", font_size=32, color=WHITE, t2c={"λ": "#FFFF00"})
        self.place_at_grid(equation, 'A4', scale_factor=1.0)
        
        # Label Lambda as Eigenvalue
        lambda_label = Text("Eigenvalue (scale factor)", font_size=20, color="#FFFF00")
        self.place_at_grid(lambda_label, 'B4', scale_factor=0.8)
        
        self.play(Write(equation))
        self.play(FadeIn(lambda_label))
        
        # Pulse the lambda in the equation
        # λ is at the 5th character index roughly in "Av = λv"
        self.play(lambda_label.animate.scale(1.2), rate_func=there_and_back)
        self.wait(2)
