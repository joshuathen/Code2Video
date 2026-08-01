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
        title = "Visual Summary: Why Convolution Matters"
        lines = [
            "Convolving distributions usually makes them smoother and wider.",
            "Square distributions convolve into triangles, then bell curves.",
            "This process reveals why the Normal distribution is common."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Color line 1
        self.play(self.lecture[0].animate.set_color(BLUE_C))
        
        # Two uniform square distributions (Rectangles)
        square1 = Rectangle(width=1.0, height=1.0, color="#FFFFFF", stroke_width=4)
        square1_label = Text("U(0, 1)", font_size=18, color="#FFFFFF")
        sq1_group = VGroup(square1, square1_label.next_to(square1, DOWN, buff=0.2))
        
        square2 = Rectangle(width=1.0, height=1.0, color="#FFFFFF", stroke_width=4)
        square2_label = Text("U(0, 1)", font_size=18, color="#FFFFFF")
        sq2_group = VGroup(square2, square2_label.next_to(square2, DOWN, buff=0.2))

        # Addressing Issue 28: Move to B4 and B6 to avoid overlap with lecture text
        self.place_at_grid(sq1_group, "B4")
        self.place_at_grid(sq2_group, "B6")

        self.play(Create(sq1_group), Create(sq2_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Color line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW_C)
        )

        # Triangle distribution
        triangle = Polygon(
            [-1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0],
            color="#FFFFFF", stroke_width=4
        )
        triangle_label = Text("Sum of 2", font_size=18, color="#FFFFFF")
        tri_group = VGroup(triangle, triangle_label.next_to(triangle, DOWN, buff=0.2))
        
        # Addressing Issue 29: Move tri_group to D4-F6 to avoid overlap
        self.place_in_area(tri_group, "D4", "F6")

        # Visualizing convolution of squares into triangle
        # Positioning plus sign and arrow relative to new positions
        plus_sign = Text("+", font_size=30).move_to(self.grid["B5"])
        
        # Arrow from between squares down towards the triangle area
        mid_point_squares = self.grid["B5"]
        tri_center = (self.grid["D4"] + self.grid["F6"]) / 2
        conv_arrow = Arrow(
            mid_point_squares + DOWN * 0.5, 
            tri_center + UP * 0.8, 
            buff=0.1, color=WHITE
        )

        self.play(Write(plus_sign))
        self.play(GrowArrow(conv_arrow))
        self.play(ReplacementTransform(VGroup(sq1_group.copy(), sq2_group.copy()), tri_group))
        self.wait(1)

        # Transition to smoother curve (Sum of 3 or 4)
        smooth_curve_pts = [
            [-1.5, 0, 0], [-0.8, 0.4, 0], [0, 1.2, 0], [0.8, 0.4, 0], [1.5, 0, 0]
        ]
        smooth_curve = VMobject(color="#FFFFFF", stroke_width=4).set_points_as_corners(smooth_curve_pts).make_smooth()
        smooth_label = Text("Sum of 3+", font_size=18, color="#FFFFFF")
        smooth_group = VGroup(smooth_curve, smooth_label.next_to(smooth_curve, DOWN, buff=0.2))
        
        # Keep smooth_group in the same area as triangle
        self.place_in_area(smooth_group, "D4", "F6")

        self.play(
            ReplacementTransform(tri_group, smooth_group),
            FadeOut(plus_sign),
            FadeOut(sq1_group),
            FadeOut(sq2_group),
            FadeOut(conv_arrow)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Color line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(GOLD)
        )

        # Final Bell Curve in Golden #FFD700
        bell_curve = FunctionGraph(
            lambda x: 1.8 * np.exp(-x**2 / 1.5),
            x_range=[-2.5, 2.5],
            color="#FFD700",
            stroke_width=6
        )
        bell_label = Text("Normal Distribution", font_size=22, color="#FFD700")
        bell_group = VGroup(bell_curve, bell_label.next_to(bell_curve, DOWN, buff=0.3))
        
        # Addressing Issue 30: Position bell_group at B4-F6 to avoid obstructing lecture text
        self.place_in_area(bell_group, "B4", "F6")

        self.play(
            ReplacementTransform(smooth_group, bell_group),
            run_time=2
        )
        
        # Final highlight
        self.play(
            bell_group.animate.scale(1.1),
            rate_func=there_and_back,
            run_time=1.5
        )

        self.wait(2)
