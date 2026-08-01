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
        # Initial Setup
        lines = [
            "Pixel feels the slope to find the way down.",
            "This slope is called the Gradient of error.",
            "A positive slope tells Pixel to move left.",
            "A negative slope means he should step right.",
            "This math tells us which direction to improve."
        ]
        self.setup_layout("Gradient Descent: Feeling the Slope", lines)

        # Create axes using the area method for layout compliance (Issue 36, 38)
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[0, 4, 1],
            axis_config={"include_tip": False, "color": GRAY},
            x_length=4,
            y_length=3
        )
        self.place_in_area(axes, "B1", "F5", scale_factor=0.85)
        
        # Parabola: f(x) = x^2
        def cost_func(x):
            return x**2
        
        # Derivative: f'(x) = 2x
        def deriv_func(x):
            return 2*x

        curve = axes.plot(cost_func, color=BLUE, x_range=[-1.8, 1.8])
        
        # State variables
        x_val = ValueTracker(1.2)
        
        # Pixel character [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/pixel.svg] (Issue 27)
        pixel = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/pixel.svg")
        pixel.set_color(RED)
        pixel.scale(0.3)
        pixel.add_updater(lambda p: p.move_to(axes.c2p(x_val.get_value(), cost_func(x_val.get_value()))))
        
        # Tangent Line
        tangent_line = Line(color="#FF3399", stroke_width=4)
        def update_tangent(line):
            x0 = x_val.get_value()
            y0 = cost_func(x0)
            slope = deriv_func(x0)
            # Line direction vector
            dx = 0.5
            p1 = axes.c2p(x0 - dx, y0 - slope * dx)
            p2 = axes.c2p(x0 + dx, y0 + slope * dx)
            line.set_points_as_corners([p1, p2])
        
        tangent_line.add_updater(update_tangent)
        
        # Descent Arrow (indicates direction to move)
        descent_arrow = Arrow(max_tip_length_to_length_ratio=0.2, color=WHITE, stroke_width=5)
        def update_arrow(arrow):
            x0 = x_val.get_value()
            y0 = cost_func(x0)
            slope = deriv_func(x0)
            # Arrow points opposite to gradient sign horizontally
            direction_x = -0.6 if slope > 0 else 0.6
            start = axes.c2p(x0, y0)
            # Arrow points slightly downward along the curve direction
            end = axes.c2p(x0 + direction_x, cost_func(x0 + direction_x))
            arrow.put_start_and_end_on(start, end)
        
        descent_arrow.add_updater(update_arrow)

        # Label Gradient (Issue 37)
        gradient_label = Text("Gradient (dJ/dw)", color=WHITE, font_size=24)
        self.place_at_grid(gradient_label, "C4", scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        # Pixel [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/pixel.svg] stands on the red point of the U-curve slope.
        self.lecture[0].set_color(RED)
        self.add(axes, curve)
        self.play(FadeIn(pixel))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # A pink #FF3399 tangent line appears showing the local slope.
        self.lecture[1].set_color("#FF3399")
        self.play(Create(tangent_line))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # A white arrow #FFFFFF indicates the direction of descent (downward).
        self.lecture[2].set_color(WHITE)
        self.play(GrowArrow(descent_arrow))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # The label 'Gradient (dJ/dw)' #FFFFFF appears near the tangent line.
        self.lecture[3].set_color(WHITE)
        self.play(Write(gradient_label))
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # Pixel [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/pixel.svg] steps down the curve, following the arrow's direction.
        self.lecture[4].set_color(WHITE)
        # Move pixel from 1.2 to 0.4
        self.play(
            x_val.animate.set_value(0.4),
            run_time=3,
            rate_func=smooth
        )
        self.wait(2)
