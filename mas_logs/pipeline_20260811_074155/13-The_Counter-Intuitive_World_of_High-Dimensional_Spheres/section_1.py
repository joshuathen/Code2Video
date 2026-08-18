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

class Section1Scene(TeachingScene):
    def construct(self):
        # Using lecture lines from storyboard
        lecture_lines = [
            "- A sphere is all points at distance r from center.",
            "- In 1D, it is two points on a line.",
            "- In 2D and 3D, we add more coordinate variables."
        ]
        self.setup_layout("Prerequisite: Defining the Boundary", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color("#00FF00")
        
        # 1D line segment highlighting distance 'r' from the center
        line_1d = NumberLine(x_range=[-2, 2, 1], length=4, color=WHITE)
        self.place_in_area(line_1d, "B2", "B5")
        
        center_dot = Dot(line_1d.number_to_point(0), color=WHITE)
        r_point = Dot(line_1d.number_to_point(1), color="#00FF00")
        r_line = Line(center_dot.get_center(), r_point.get_center(), color="#00FF00", stroke_width=6)
        r_label = MathTex("r", color="#00FF00")
        self.place_at_grid(r_label, "A4", scale_factor=1.0)
        
        self.play(Create(line_1d), FadeIn(center_dot))
        self.play(Create(r_line), FadeIn(r_point), Write(r_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Reset color of line 1, highlight line 2
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        # Transition to a 2D circle with equation x² + y² = r² appearing in #FFFF00
        # Clean up 1D visuals
        self.play(FadeOut(line_1d, center_dot, r_point, r_line, r_label))
        
        circle_2d = Circle(radius=1.0, color="#FFFF00")
        self.place_in_area(circle_2d, "C2", "E4")
        
        eq_2d = MathTex("x^2 + y^2 = r^2", color="#FFFF00")
        self.place_at_grid(eq_2d, "D6", scale_factor=0.9)
        
        self.play(Create(circle_2d), Write(eq_2d))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Reset color of line 2, highlight line 3
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        
        # Expand the 2D circle into a 3D sphere, adding the z² term to the equation in #00FFFF
        # Representing 3D with a circle and an elliptical "equator"
        sphere_visual = VGroup(
            Circle(radius=1.0, color="#00FFFF"),
            Ellipse(width=2.0, height=0.4, color="#00FFFF").set_stroke(opacity=0.6)
        )
        self.place_in_area(sphere_visual, "C2", "E4")
        
        eq_3d = MathTex("x^2 + y^2 + z^2 = r^2", color="#00FFFF")
        self.place_at_grid(eq_3d, "D6", scale_factor=0.9)
        
        self.play(
            ReplacementTransform(circle_2d, sphere_visual[0]),
            FadeIn(sphere_visual[1]),
            ReplacementTransform(eq_2d, eq_3d)
        )
        self.wait(3)
