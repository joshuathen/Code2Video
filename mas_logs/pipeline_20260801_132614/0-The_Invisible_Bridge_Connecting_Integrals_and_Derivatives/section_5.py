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
        # Using simple Text instead of MathTex to avoid LaTeX compilation issues (B008)
        title = "Visualizing the Connection: The Area Function"
        lines = [
            "Let's define a function for the accumulating area.",
            "As we move right, the total area grows.",
            "The rate of growth depends on the function's height.",
            "Taller curves add area faster than shorter ones.",
            "The derivative of area is the original function."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_CURVE = WHITE
        COLOR_AREA = "#FFFF00"
        COLOR_DOT = "#FF0000"
        COLOR_TEXT = "#00FFFF"

        # Axes setup
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 5, 1],
            axis_config={"include_tip": True, "color": GREY},
            x_length=5,
            y_length=4
        )
        # Fix for Issue #41: Adjusted area to prevent overlap with ftc_formula
        self.place_in_area(axes, 'B1', 'E6', scale_factor=0.8)
        
        # Curve f(t) = 0.1t^2 + 1
        func = lambda t: 0.1 * t**2 + 1
        curve = axes.plot(func, x_range=[0, 5.5], color=COLOR_CURVE)
        
        # Use Text instead of MathTex (B008)
        curve_label = Text("f(t)", font_size=20, color=COLOR_CURVE)
        self.place_at_grid(curve_label, "B5", scale_factor=1.0)
        
        # ValueTracker for x
        x_tracker = ValueTracker(1.0)

        # === Animation for Lecture Line 1 ===
        # Let's define a function for the accumulating area.
        self.lecture[0].set_color(COLOR_CURVE)
        
        # Sliding vertical line
        vline = always_redraw(lambda: axes.get_vertical_line(
            axes.c2p(x_tracker.get_value(), func(x_tracker.get_value())),
            color=WHITE,
            stroke_width=2
        ))
        
        # Pre-create text to avoid recreation in always_redraw
        x_label = Text("x", font_size=20, color=WHITE)
        x_label.add_updater(lambda m: m.next_to(vline, DOWN, buff=0.1))

        self.play(Create(axes), Create(curve), Write(curve_label))
        self.play(Create(vline), FadeIn(x_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # As we move right, the total area grows.
        self.lecture[1].set_color(COLOR_AREA)
        
        area = always_redraw(lambda: axes.get_area(
            curve,
            x_range=[0, x_tracker.get_value()],
            color=COLOR_AREA,
            opacity=0.4
        ))
        
        area_label = Text("A(x) = Area from 0 to x", font_size=20, color=COLOR_AREA)
        # Fix for Issue #42: Better centering for long label
        self.place_in_area(area_label, 'A2', 'A5', scale_factor=0.8)

        self.add(area)
        self.play(Write(area_label))
        self.play(x_tracker.animate.set_value(4.5), run_time=3, rate_func=rate_functions.linear)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The rate of growth depends on the function's height.
        self.lecture[2].set_color(COLOR_DOT)
        
        # Use simple shapes and updates
        dot = always_redraw(lambda: Dot(
            axes.c2p(x_tracker.get_value(), func(x_tracker.get_value())),
            color=COLOR_DOT
        ))
        
        height_line = always_redraw(lambda: Line(
            axes.c2p(x_tracker.get_value(), 0),
            axes.c2p(x_tracker.get_value(), func(x_tracker.get_value())),
            color=COLOR_DOT,
            stroke_width=4
        ))
        
        height_label = Text("f(x)", font_size=20, color=COLOR_DOT)
        height_label.add_updater(lambda m: m.next_to(dot, UP, buff=0.1))

        self.play(Create(dot), Create(height_line), FadeIn(height_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Taller curves add area faster than shorter ones.
        self.lecture[3].set_color(COLOR_TEXT)
        
        # Demonstrate growth by moving x again
        self.play(x_tracker.animate.set_value(2.0), run_time=2)
        self.play(x_tracker.animate.set_value(5.0), run_time=3)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The derivative of area is the original function.
        self.lecture[4].set_color(WHITE)
        
        # Rate of Area Growth = f(x)
        ftc_formula = Text("Rate of Area Growth = f(x)", font_size=24, color=WHITE)
        # Fix for Issue #43: Centered long label at bottom
        self.place_in_area(ftc_formula, 'F2', 'F5', scale_factor=0.8)
        
        self.play(Write(ftc_formula))
        self.play(
            Indicate(ftc_formula, color=COLOR_DOT),
            Indicate(height_line, color=COLOR_DOT),
            Indicate(height_label, color=COLOR_DOT),
            run_time=2
        )
        self.wait(2)
