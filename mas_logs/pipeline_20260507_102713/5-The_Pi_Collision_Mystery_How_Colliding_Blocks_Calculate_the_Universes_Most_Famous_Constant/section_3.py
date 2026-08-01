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
        # Setup content
        title = "Geometric Transformation: From Ellipse to Circle"
        lines = [
            'Kinetic energy conservation defines an elliptical velocity boundary.',
            'Scaling the velocity morphs the ellipse into a circle.',
            "This circular geometry simplifies the system's complex dynamics."
        ]
        self.setup_layout(title, lines)

        # Colors
        ELLIPSE_COLOR = "#0000FF"
        CIRCLE_COLOR = "#00FF00"
        EQUATION_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Create Axes - Positioned at B3-F6, scale 0.7 (Issue 31)
        axes = Axes(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True}
        )
        self.place_in_area(axes, "B3", "F6", scale_factor=0.7)
        
        v1_label = Text("v₁", font_size=24).next_to(axes.x_axis.get_end(), RIGHT, buff=0.1)
        v2_label = Text("v₂", font_size=24).next_to(axes.y_axis.get_top(), UP, buff=0.1)
        
        # Initial Ellipse: m*v1^2 + M*v2^2 = const
        ellipse = Ellipse(width=3.6 * 0.7, height=1.0 * 0.7, color=ELLIPSE_COLOR)
        ellipse.move_to(axes.c2p(0, 0))
        
        self.play(Create(axes), Write(v1_label), Write(v2_label))
        self.play(Create(ellipse))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Morphing to circle
        self.play(
            ellipse.animate.stretch(3.6, dim=1).set_color(CIRCLE_COLOR),
            run_time=2
        )
        circle = ellipse 
        
        # Display Equations - Corrected placement (Issue 29, 30)
        eq1 = Text("v₁² + [√(M/m)v₂]² = C", font_size=22, color=EQUATION_COLOR)
        eq2 = Text("x² + y² = R²", font_size=28, color=EQUATION_COLOR)
        
        self.place_in_area(eq1, "A1", "A2", scale_factor=0.9)
        self.place_in_area(eq2, "B1", "B2", scale_factor=0.9)
        
        self.play(Write(eq1))
        self.play(ReplacementTransform(eq1.copy(), eq2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # State Point and Circular Motion
        dot = Dot(color=YELLOW)
        dot.move_to(circle.point_at_angle(PI/4))
        dot_label = Text("(x, y)", font_size=24, color=YELLOW).next_to(dot, UR, buff=0.1)
        
        self.play(FadeIn(dot), Write(dot_label))
        
        theta = ValueTracker(PI/4)
        dot.add_updater(lambda d: d.move_to(circle.point_at_angle(theta.get_value())))
        dot_label.add_updater(lambda l: l.next_to(dot, UR, buff=0.1))
        
        self.play(theta.animate.set_value(PI/4 + 2*PI), run_time=4, rate_func=linear)
        
        # Final highlight
        self.play(circle.animate.set_stroke(width=8), run_time=0.5)
        self.play(circle.animate.set_stroke(width=4), run_time=0.5)
        
        self.wait(2)
        
        # Cleanup updaters
        dot.clear_updaters()
        dot_label.clear_updaters()
