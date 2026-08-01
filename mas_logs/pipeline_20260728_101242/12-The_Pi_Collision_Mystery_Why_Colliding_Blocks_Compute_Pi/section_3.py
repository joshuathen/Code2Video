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
        self.setup_layout("The Energy Ellipse to Energy Circle", [
            "Conservation of energy defines an elliptical path.",
            "We rescale the axes based on mass ratios.",
            "This transformation turns the ellipse into a circle."
        ])

        # Colors
        CYAN = "#00FFFF"
        YELLOW = "#FFFF00"
        MAGENTA = "#FF00FF"

        # === Animation for Lecture Line 1 ===
        # Line 1: Conservation of energy defines an elliptical path.
        self.play(self.lecture[0].animate.set_color(CYAN))
        
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE}
        )
        # Issue 27: Re-adjust main axes group to occupy 'B2' to 'E6'
        self.place_in_area(axes, 'B2', 'E6', scale_factor=0.8)
        
        # Issue 25: Horizontal label at 'D6'
        label_v1 = MathTex("v_1", font_size=24)
        self.place_at_grid(label_v1, 'D6', scale_factor=0.7)
        
        # Issue 26: Vertical label at 'A4'
        label_v2 = MathTex("v_2", font_size=24)
        self.place_at_grid(label_v2, 'A4', scale_factor=0.7)
        
        # Initial ellipse
        # Scene units are used for width/height. Axes scaling is handled by place_in_area.
        ellipse = Ellipse(width=1.5, height=3.0, color=CYAN, stroke_width=4)
        ellipse.move_to(axes.c2p(0, 0))
        
        self.play(Create(axes), Write(label_v1), Write(label_v2))
        self.play(Create(ellipse))
        # Highlight to show mathematical complexity
        self.play(ellipse.animate.set_stroke(width=8), run_time=0.4)
        self.play(ellipse.animate.set_stroke(width=4), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: We rescale the axes based on mass ratios.
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(CYAN))
        
        # Rescaled labels
        label_v1_rescaled = MathTex("v_1\\sqrt{m}", font_size=24)
        self.place_at_grid(label_v1_rescaled, 'D6', scale_factor=0.7)
        
        label_v2_rescaled = MathTex("v_2\\sqrt{M}", font_size=24)
        self.place_at_grid(label_v2_rescaled, 'A4', scale_factor=0.7)
        
        self.play(
            Transform(label_v1, label_v1_rescaled),
            Transform(label_v2, label_v2_rescaled)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: This transformation turns the ellipse into a circle.
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        # Circle radius
        circle = Circle(radius=1.5, color=YELLOW, stroke_width=4)
        circle.move_to(axes.c2p(0, 0))
        
        self.play(Transform(ellipse, circle))
        
        # State dot
        dot = Dot(color=MAGENTA).move_to(circle.point_at_angle(0))
        self.play(FadeIn(dot))
        
        # Show state dot moving along the circle's perimeter.
        self.play(MoveAlongPath(dot, circle), run_time=4, rate_func=linear)
        self.wait(2)
