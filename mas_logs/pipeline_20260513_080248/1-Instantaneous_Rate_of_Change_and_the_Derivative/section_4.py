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
        # Setup lines and layout
        lecture_lines_text = [
            'At the limit, the secant line becomes the tangent.',
            'Magnified, the curve appears to be a straight line.',
            'The tangent line touches the curve at one point.',
            'Its slope represents the instantaneous rate of change.',
            'This unique line defines the derivative at this point.'
        ]
        self.setup_layout("The Birth of the Tangent Line", lecture_lines_text)

        # Colors
        TANGENT_COLOR = "#2ECC71"  # Green
        CURVE_COLOR = "#3498DB"    # Blue
        POINT_COLOR = WHITE

        # --- Base Graphic Elements ---
        # Axes and Curve
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": GRAY}
        )
        self.place_in_area(axes, "C2", "F5")
        
        def func(x):
            return 0.3 * x**2 + 0.5
            
        graph = axes.plot(func, color=CURVE_COLOR)
        
        # Points A and B
        x_a = 1.2
        x_b_init = 3.0
        p_a = Dot(axes.c2p(x_a, func(x_a)), color=POINT_COLOR, radius=0.08)
        p_b = Dot(axes.c2p(x_b_init, func(x_b_init)), color=POINT_COLOR, radius=0.08)
        
        label_a = Text("A", font_size=18).next_to(p_a, LEFT, buff=0.1)
        label_b = Text("B", font_size=18).next_to(p_b, RIGHT, buff=0.1)
        
        # Secant Line
        secant = Line(
            axes.c2p(0.5, func(x_a) + (func(x_b_init) - func(x_a))/(x_b_init - x_a) * (0.5 - x_a)),
            axes.c2p(3.5, func(x_a) + (func(x_b_init) - func(x_a))/(x_b_init - x_a) * (3.5 - x_a)),
            color=WHITE,
            stroke_width=2
        )

        # h bracket (visualized as a simple line with tip)
        h_line = Line(axes.c2p(x_a, 0.4), axes.c2p(x_b_init, 0.4), color=WHITE).add_tip(tip_length=0.1)
        h_label = Text("h", font_size=18).next_to(h_line, DOWN, buff=0.1)
        h_group = VGroup(h_line, h_label)

        self.add(axes, graph, p_a, p_b, label_a, label_b, secant, h_group)

        # === Animation for Lecture Line 1 ===
        # "At the limit, the secant line becomes the tangent."
        self.play(self.lecture[0].animate.set_color(TANGENT_COLOR))
        
        # Target tangent slope calculation
        slope_a = 0.6 * x_a 
        x_b_limit = 1.22
        target_p_b_pos = axes.c2p(x_b_limit, func(x_b_limit))
        
        tangent_line_target = Line(
            axes.c2p(x_a - 1.2, func(x_a) - slope_a * 1.2),
            axes.c2p(x_a + 1.2, func(x_a) + slope_a * 1.2),
            color=TANGENT_COLOR,
            stroke_width=3
        )
        
        tangent_label = Text("Tangent Line", font_size=20, color=TANGENT_COLOR)
        # Issue 37: Fix positioning of tangent_label
        self.place_at_grid(tangent_label, "B4", scale_factor=0.8)

        self.play(
            p_b.animate.move_to(target_p_b_pos),
            label_b.animate.move_to(target_p_b_pos + RIGHT*0.2),
            secant.animate.become(tangent_line_target).set_color(TANGENT_COLOR),
            h_group.animate.scale(0.1).set_opacity(0),
            run_time=2
        )
        self.play(FadeIn(tangent_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Magnified, the curve appears to be a straight line."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(TANGENT_COLOR)
        )
        
        # Issue 25: Use magnifier asset
        magnifier = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/magnifier.svg")
        # Issue 38: Fix positioning and scale of zoom circle/magnifier
        self.place_at_grid(magnifier, "D2", scale_factor=0.6)
        
        # Zoom content: Curve and tangent look like lines
        zoom_line_curve = Line(LEFT, RIGHT, color=CURVE_COLOR).scale(0.4).rotate(slope_a * 0.4)
        zoom_line_tangent = Line(LEFT, RIGHT, color=TANGENT_COLOR).scale(0.4).rotate(slope_a * 0.4).shift(UP*0.04)
        zoom_content = VGroup(zoom_line_curve, zoom_line_tangent).move_to(magnifier.get_center() + UP*0.1)
        
        connector = Line(p_a.get_center(), magnifier.get_center(), stroke_width=1, color=GRAY, buff=0.1)
        
        self.play(Create(magnifier), Create(connector))
        self.play(FadeIn(zoom_content))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "The tangent line touches the curve at one point."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(TANGENT_COLOR)
        )
        
        # Remove B and h
        self.play(
            FadeOut(p_b), 
            FadeOut(label_b), 
            FadeOut(h_group),
            FadeOut(magnifier),
            FadeOut(connector),
            FadeOut(zoom_content)
        )
        
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Its slope represents the instantaneous rate of change."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(TANGENT_COLOR)
        )
        
        slope_formula = Text("Slope of Tangent = Instantaneous Rate", font_size=24, color=TANGENT_COLOR)
        # Issue 36: Fix slope_formula position and scale
        self.place_in_area(slope_formula, "A1", "A5", scale_factor=0.7)
        
        self.play(Write(slope_formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This unique line defines the derivative at this point."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(TANGENT_COLOR)
        )
        
        # Tangent line flashes twice
        self.play(secant.animate.set_stroke(width=8), run_time=0.25)
        self.play(secant.animate.set_stroke(width=3), run_time=0.25)
        self.play(secant.animate.set_stroke(width=8), run_time=0.25)
        self.play(secant.animate.set_stroke(width=3), run_time=0.25)
        
        self.play(Indicate(p_a, color=TANGENT_COLOR))
        
        self.wait(2)
