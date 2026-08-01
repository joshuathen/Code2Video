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
        # Section details based on Stage-3 Prompt
        lecture_lines = [
            '1D roots occur where functions cross the x-axis.',
            '2D roots are intersections of two zero-level curves.',
            'Simple bisection fails because 2D paths are more complex.'
        ]
        self.setup_layout("The Challenge: From 1D to 2D Root Finding", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Color current lecture line
        self.play(self.lecture[0].animate.set_color("#FFFF00"))
        
        # Draw a 1D axis and a curve y = x - 2 crossing it at x=2 with a highlighted dot (#FFFF00)
        axis_1d = Line(LEFT*2.5, RIGHT*2.5, color=WHITE)
        # Visually represent a crossing at the center of the local scene
        curve_1d = Line(LEFT*1.5 + DOWN*1.5, RIGHT*1.5 + UP*1.5, color="#FFFF00")
        root_dot = Dot(ORIGIN, color="#FFFF00")
        root_label = Text("x = 2", font_size=18, color="#FFFF00").next_to(root_dot, UR, buff=0.1)
        
        scene_1d = VGroup(axis_1d, curve_1d, root_dot, root_label)
        # Resolved Issue 26: Positioning scene_1d in B2-D5
        self.place_in_area(scene_1d, "B2", "D5", scale_factor=1.0)
        
        self.play(Create(axis_1d), Create(curve_1d))
        self.play(FadeIn(root_dot), Write(root_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        self.play(FadeOut(scene_1d))
        
        # Transition to 2D: Two intersecting curves u(x,y)=0 (Blue) and v(x,y)=0 (Red)
        axes_2d = VGroup(
            Line(LEFT*2.2, RIGHT*2.2, color=WHITE),
            Line(DOWN*1.7, UP*1.7, color=WHITE)
        )
        u_curve = Line(LEFT*1.5 + DOWN*1, RIGHT*1.5 + UP*1, color="#0000FF") # Blue
        v_curve = Line(LEFT*1.5 + UP*1, RIGHT*1.5 + DOWN*1, color="#FF0000") # Red
        u_label = Text("u(x,y)=0", font_size=16, color="#0000FF").next_to(u_curve.get_end(), RIGHT, buff=0.1)
        v_label = Text("v(x,y)=0", font_size=16, color="#FF0000").next_to(v_curve.get_start(), LEFT, buff=0.1)
        
        root_dot_2d = Dot(ORIGIN, color=WHITE)
        root_label_2d = Text("f(x,y) = (0,0)", font_size=16).next_to(root_dot_2d, UR, buff=0.1)
        
        scene_2d = VGroup(axes_2d, u_curve, v_curve, u_label, v_label, root_dot_2d, root_label_2d)
        # Resolved Issue 26: Positioning scene_2d in B2-D5
        self.place_in_area(scene_2d, "B2", "D5", scale_factor=0.9)
        
        self.play(Create(axes_2d))
        self.play(Create(u_curve), Create(v_curve), Write(u_label), Write(v_label))
        self.play(FadeIn(root_dot_2d), Write(root_label_2d))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FF5555")
        )
        # Transition existing scene to make room for bisection logic
        self.play(scene_2d.animate.scale(0.6).shift(UP*1.2))
        
        # 1D Bisection Visualization
        line_seg = Line(LEFT*0.8, RIGHT*0.8, color=WHITE)
        bracket_l = Text("[", font_size=20).move_to(line_seg.get_left())
        bracket_r = Text("]", font_size=20).move_to(line_seg.get_right())
        arrow_l = Arrow(ORIGIN, LEFT*0.7, buff=0, color=YELLOW, stroke_width=2)
        arrow_r = Arrow(ORIGIN, RIGHT*0.7, buff=0, color=YELLOW, stroke_width=2)
        label_1d = Text("1D: 2 Directions", font_size=14).next_to(line_seg, DOWN)
        bisection_1d = VGroup(line_seg, bracket_l, bracket_r, arrow_l, arrow_r, label_1d)
        # Resolved Issue 27: Position E2, Scale 1.2
        self.place_at_grid(bisection_1d, "E2", scale_factor=1.2)
        
        # 2D Bisection using Asset
        # Resolved Issue 22: Using SVGMobject for [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/square.svg]
        square_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/square.svg").set_color(WHITE).scale(0.5)
        arrows_2d = VGroup(*[
            Arrow(ORIGIN, v*0.6, buff=0, color=RED, stroke_width=2) 
            for v in [UP, DOWN, LEFT, RIGHT, UR, DL, UL, DR]
        ])
        label_2d = Text("2D: Ambiguous", font_size=14).next_to(square_asset, DOWN)
        bisection_2d = VGroup(square_asset, arrows_2d, label_2d)
        # Resolved Issue 27: Position E5, Scale 1.2
        self.place_at_grid(bisection_2d, "E5", scale_factor=1.2)
        
        self.play(FadeIn(bisection_1d), FadeIn(bisection_2d))
        self.wait(3)
