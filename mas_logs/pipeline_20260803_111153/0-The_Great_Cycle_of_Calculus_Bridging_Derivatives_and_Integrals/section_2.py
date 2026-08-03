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

class Section2Scene(TeachingScene):
    def construct(self):
        # Data
        title_text = "Prerequisite Recap: The Slope of Change"
        lecture_lines = [
            "Differentiation helps us calculate the rate of change.",
            "Graphically, the derivative is the slope of the tangent.",
            "It transforms a position function into a velocity function."
        ]
        
        # Setup Layout
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        POSITION_COLOR = "#ADD8E6"
        VELOCITY_COLOR = "#FFA500"
        HIGHLIGHT_COLOR = YELLOW

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Axes setup (Issue 34: place_in_area B3 to E6)
        axes = Axes(
            x_range=[0, 5],
            y_range=[0, 20, 5],
            x_length=4.5,
            y_length=3.5,
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.place_in_area(axes, 'B3', 'E6')
        
        x_label = Text("Time (t)", font_size=16).next_to(axes.x_axis, DOWN, buff=0.2)
        y_label = Text("Position (s)", font_size=16).rotate(90 * DEGREES).next_to(axes.y_axis, LEFT, buff=0.2)
        
        # Position curve: s(t) = t^2
        curve = axes.plot(lambda t: t**2, x_range=[0, 4], color=POSITION_COLOR)
        
        # Hiker Asset (Issue 30)
        hiker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hiker.svg", height=0.5)
        hiker.set_color(POSITION_COLOR)
        # Place hiker at the start of the curve
        hiker.move_to(axes.c2p(0, 0))
        
        self.play(Create(axes), Create(x_label), Create(y_label))
        self.play(Create(curve), FadeIn(hiker))
        
        # Animate hiker along the curve
        self.play(MoveAlongPath(hiker, curve), run_time=2, rate_func=linear)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Focus on a point (t=2.5)
        t_val = 2.5
        point = axes.c2p(t_val, t_val**2)
        dot = Dot(point, color=WHITE)
        
        # Tangent line at t=2.5: s'(t) = 2t. s'(2.5) = 5.
        # Equation: y - 6.25 = 5(x - 2.5) => y = 5x - 6.25
        tangent_line = axes.plot(lambda t: 5 * t - 6.25, x_range=[1.5, 3.5], color=WHITE)
        
        # Zooming lens (Circle)
        lens = Circle(radius=0.6, color=WHITE, stroke_width=2).move_to(point)
        
        self.play(Create(lens))
        self.play(FadeIn(dot), Create(tangent_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Color tangent line and label it
        self.play(tangent_line.animate.set_color(VELOCITY_COLOR))
        
        # Label for derivative
        velocity_label = Text("Velocity (Derivative)", font_size=18, color=VELOCITY_COLOR)
        self.place_at_grid(velocity_label, "B5")
        
        # Arrow pointing from tangent to label
        arrow = Arrow(
            start=velocity_label.get_bottom(),
            end=tangent_line.get_center(),
            color=VELOCITY_COLOR,
            buff=0.1,
            stroke_width=2,
            max_tip_length_to_length_ratio=0.1
        )
        
        self.play(Write(velocity_label), Create(arrow))
        self.wait(2)
