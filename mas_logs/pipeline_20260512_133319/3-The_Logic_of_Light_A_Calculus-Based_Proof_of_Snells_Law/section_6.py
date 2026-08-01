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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize Scene
        self.setup_layout(
            "Connecting Geometry to Trigonometry", 
            [
                'Look back at our geometric triangles.', 
                'These ratios are sines of the angles.', 
                'Substitute sines into our calculus result.'
            ]
        )
        
        # Colors
        COLOR_T1 = BLUE_B
        COLOR_T2 = GREEN_B
        COLOR_EQ = WHITE
        COLOR_FINAL = "#FFD700" # Gold
        
        # === Animation for Lecture Line 1 ===
        # Highlight the lecture line
        self.lecture[0].set_color(YELLOW)
        
        # Geometry for Triangle 1
        t1_top = np.array([0, 1.2, 0])
        t1_bottom = np.array([0, 0, 0])
        t1_right = np.array([0.9, 0, 0])
        
        tri1 = Polygon(t1_top, t1_bottom, t1_right, color=COLOR_T1, stroke_width=2)
        side_x1 = Line(t1_bottom, t1_right, color=COLOR_T1, stroke_width=5)
        hyp_d1 = Line(t1_top, t1_right, color=COLOR_T1, stroke_width=5)
        label_x1 = Text("x", font_size=24).next_to(side_x1, DOWN, buff=0.1)
        label_d1 = Text("d\u2081", font_size=24).move_to(hyp_d1.get_center() + UP*0.3 + RIGHT*0.3)
        label_theta1 = Text("\u03B8\u2081", font_size=24).move_to(t1_top + DOWN*0.4 + RIGHT*0.2)
        
        group_t1 = VGroup(tri1, side_x1, hyp_d1, label_x1, label_d1, label_theta1)
        # Issue 44: utilize A1-B3 for group_t1
        self.place_in_area(group_t1, 'A1', 'B3', scale_factor=0.8)
        
        # Geometry for Triangle 2
        t2_left = np.array([0, 0, 0])
        t2_right = np.array([1.1, 0, 0])
        t2_bottom = np.array([1.1, -1.0, 0])
        
        tri2 = Polygon(t2_left, t2_right, t2_bottom, color=COLOR_T2, stroke_width=2)
        side_x2 = Line(t2_left, t2_right, color=COLOR_T2, stroke_width=5)
        hyp_d2 = Line(t2_left, t2_bottom, color=COLOR_T2, stroke_width=5)
        label_x2 = Text("w - x", font_size=24).next_to(side_x2, UP, buff=0.1)
        label_d2 = Text("d\u2082", font_size=24).move_to(hyp_d2.get_center() + DOWN*0.3 + LEFT*0.3)
        label_theta2 = Text("\u03B8\u2082", font_size=24).move_to(t2_bottom + UP*0.4 + LEFT*0.2)

        group_t2 = VGroup(tri2, side_x2, hyp_d2, label_x2, label_d2, label_theta2)
        # Issue 45: utilize C1-D3 for group_t2
        self.place_in_area(group_t2, 'C1', 'D3', scale_factor=0.8)
        
        self.play(FadeIn(group_t1), FadeIn(group_t2))
        self.wait(0.5)
        
        # Pulse x and d1
        self.play(
            side_x1.animate.set_stroke(width=12),
            hyp_d1.animate.set_stroke(width=12),
            run_time=0.6, rate_func=there_and_back
        )
        self.play(
            side_x1.animate.set_stroke(width=5),
            hyp_d1.animate.set_stroke(width=5),
            run_time=0.4
        )
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # sin(theta1) = x/d1
        sin1_eq = VGroup(
            Text("sin(\u03B8\u2081) = ", font_size=28),
            VGroup(
                Text("x", font_size=28),
                Line(LEFT*0.3, RIGHT*0.3, stroke_width=2),
                Text("d\u2081", font_size=28)
            ).arrange(DOWN, buff=0.1)
        ).arrange(RIGHT, buff=0.15)
        
        # sin(theta2) = (w-x)/d2
        sin2_eq = VGroup(
            Text("sin(\u03B8\u2082) = ", font_size=28),
            VGroup(
                Text("w - x", font_size=28),
                Line(LEFT*0.5, RIGHT*0.5, stroke_width=2),
                Text("d\u2082", font_size=28)
            ).arrange(DOWN, buff=0.1)
        ).arrange(RIGHT, buff=0.15)
        
        # Issue 44: utilize A4-B6 for sin1_eq
        self.place_in_area(sin1_eq, 'A4', 'B6', scale_factor=0.9)
        # Issue 45: utilize C4-D6 for sin2_eq
        self.place_in_area(sin2_eq, 'C4', 'D6', scale_factor=0.9)
        
        self.play(Write(sin1_eq), Write(sin2_eq))
        self.wait(1.5)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Substitute into sin(theta1)/v1 = sin(theta2)/v2
        final_eq = VGroup(
            VGroup(
                Text("sin(\u03B8\u2081)", font_size=32),
                Line(LEFT*0.6, RIGHT*0.6, stroke_width=2),
                Text("v\u2081", font_size=32)
            ).arrange(DOWN, buff=0.1),
            Text(" = ", font_size=32),
            VGroup(
                Text("sin(\u03B8\u2082)", font_size=32),
                Line(LEFT*0.6, RIGHT*0.6, stroke_width=2),
                Text("v\u2082", font_size=32)
            ).arrange(DOWN, buff=0.1)
        ).arrange(RIGHT, buff=0.3).set_color(COLOR_FINAL)
        
        # Issue 46: utilize E3-F5 for final_eq
        self.place_in_area(final_eq, 'E3', 'F5', scale_factor=1.0)
        
        # Visual substitution transition
        self.play(
            ReplacementTransform(sin1_eq.copy(), final_eq[0]),
            ReplacementTransform(sin2_eq.copy(), final_eq[2]),
            FadeIn(final_eq[1])
        )
        self.wait(2.5)
