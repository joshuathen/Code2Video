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
        # 1. Setup layout using storyboard data
        title = "The Secant Line Bridge"
        lecture_lines = [
            "A secant line connects two points on a curve.",
            "Its slope approximates the speed between these two moments.",
            "But this average is still just a blurry guess."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors from storyboard
        COLOR_CURVE = "#FFFFFF"
        COLOR_POINT_Q = "#00FF00"
        COLOR_SECANT = "#FFFF00"
        
        # 2. Prepare graph elements
        axes = Axes(
            x_range=[0, 3, 1],
            y_range=[0, 5, 1],
            axis_config={"include_tip": True, "include_numbers": False},
            x_length=4,
            y_length=4,
            tips=True
        ).set_color(GRAY)
        
        # Curve y = x^2
        curve = axes.plot(lambda x: x**2, x_range=[0, 2.2], color=COLOR_CURVE)
        
        # Point P at (1,1)
        p_coords = axes.c2p(1, 1)
        point_p = Dot(p_coords, color=COLOR_CURVE)
        label_p = MathTex("P(1,1)", font_size=20, color=COLOR_CURVE).next_to(point_p, DOWN + RIGHT, buff=0.1)
        
        # Point Q at (2,4)
        q_coords = axes.c2p(2, 4)
        point_q = Dot(q_coords, color=COLOR_POINT_Q)
        label_q = MathTex("Q(2,4)", font_size=20, color=COLOR_POINT_Q).next_to(point_q, UP + LEFT, buff=0.1)
        
        # Secant Line connecting P and Q
        secant_line = Line(p_coords, q_coords, color=COLOR_SECANT)
        
        # Slope Label
        slope_label = MathTex(r"\text{Slope} = 3", font_size=22, color=COLOR_SECANT)
        slope_label.next_to(secant_line.get_center(), RIGHT, buff=0.2)

        # Group and position - addressing Issue 28:
        # Move from A1 to A2 and scale from 0.9 to 0.8 to avoid collision with title.
        graph_group = VGroup(axes, curve, point_p, label_p, point_q, label_q, secant_line, slope_label)
        self.place_in_area(graph_group, "A2", "F6", scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        # "A secant line connects two points on a curve."
        self.play(self.lecture[0].animate.set_color(COLOR_CURVE))
        self.play(Create(axes), Create(curve))
        self.play(FadeIn(point_p), Write(label_p))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Its slope approximates the speed between these two moments."
        self.play(self.lecture[1].animate.set_color(COLOR_POINT_Q))
        self.play(FadeIn(point_q), Write(label_q))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "But this average is still just a blurry guess."
        self.play(self.lecture[2].animate.set_color(COLOR_SECANT))
        self.play(Create(secant_line))
        self.play(Write(slope_label))
        self.wait(2)
