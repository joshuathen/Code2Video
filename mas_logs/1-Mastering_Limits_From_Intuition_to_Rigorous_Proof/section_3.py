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
        # Initial layout setup
        self.setup_layout("The Indeterminate Form Crisis", [
            "Direct substitution can lead to indeterminate forms like 0/0.",
            "This represents a tug-of-war between numerator and denominator.",
            "Visually, it looks like a black hole of uncertainty."
        ])

        # Colors
        RED_COLOR = "#FF0000"
        CYAN_COLOR = "#00FFFF"
        ORANGE_COLOR = "#FFA500"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(RED_COLOR))
        
        # Formula: f(x) = sin(x)/x (Using Text instead of MathTex to avoid LaTeX dependency)
        func_text = Text("f(x) = sin(x) / x", color=RED_COLOR)
        self.place_in_area(func_text, "A1", "A3", scale_factor=0.8)
        
        # Result: f(0) = 0/0
        result_text = Text("f(0) = 0 / 0", color=RED_COLOR)
        self.place_in_area(result_text, "A4", "A6", scale_factor=0.8)
        
        # Axes and pulsing circle at origin
        axes_1 = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1.5, 1.5, 1],
            axis_config={"include_tip": False, "color": WHITE}
        )
        self.place_in_area(axes_1, "C2", "F5", scale_factor=0.6)
        
        pulsing_circle = Circle(radius=0.15, color=RED_COLOR, fill_opacity=0.5)
        pulsing_circle.move_to(axes_1.c2p(0, 0))
        
        self.play(Write(func_text), Write(result_text))
        self.play(Create(axes_1))
        self.play(FadeIn(pulsing_circle))
        
        # Pulse animation (Repeated to simulate iteration_count)
        for _ in range(2):
            self.play(
                pulsing_circle.animate.scale(1.5).set_fill(opacity=0.8),
                rate_func=there_and_back,
                run_time=1.5
            )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Clear previous for split screen
        self.play(
            FadeOut(func_text), FadeOut(result_text), FadeOut(axes_1), FadeOut(pulsing_circle),
            self.lecture[0].animate.set_color(WHITE)
        )
        self.play(self.lecture[1].animate.set_color(CYAN_COLOR))

        # Split screen components
        # Left side: y = sin(x)
        axes_sin = Axes(x_range=[-3, 3], y_range=[-1.5, 1.5], axis_config={"color": WHITE}).scale(0.4)
        self.place_in_area(axes_sin, "B1", "E3")
        sin_graph = axes_sin.plot(lambda x: np.sin(x), color=CYAN_COLOR)
        sin_label = Text("sin(x)", color=CYAN_COLOR).scale(0.6)
        self.place_at_grid(sin_label, "A2")

        # Right side: y = x
        axes_x = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"color": WHITE}).scale(0.4)
        self.place_in_area(axes_x, "B4", "E6")
        linear_graph = axes_x.plot(lambda x: x, color=ORANGE_COLOR)
        linear_label = Text("x", color=ORANGE_COLOR).scale(0.6)
        self.place_at_grid(linear_label, "A5")

        self.play(
            Create(axes_sin), Create(sin_graph), Write(sin_label),
            Create(axes_x), Create(linear_graph), Write(linear_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Clear and transition
        self.play(
            FadeOut(axes_sin), FadeOut(sin_graph), FadeOut(sin_label),
            FadeOut(axes_x), FadeOut(linear_graph), FadeOut(linear_label),
            self.lecture[1].animate.set_color(WHITE)
        )
        self.play(self.lecture[2].animate.set_color(WHITE_COLOR))

        # Main graph: sin(x)/x
        axes_final = Axes(
            x_range=[-10, 10, 2],
            y_range=[-0.5, 1.5, 0.5],
            axis_config={"include_tip": True, "color": WHITE}
        )
        self.place_in_area(axes_final, "B2", "F5", scale_factor=0.7)
        
        # Handle the point at x=0 for sin(x)/x
        graph_final = axes_final.plot(
            lambda x: np.sin(x)/x if abs(x) > 0.001 else 1.0,
            color=WHITE_COLOR,
            use_smoothing=True
        )
        
        limit_dot = Dot(color=CYAN_COLOR)
        limit_dot.move_to(axes_final.c2p(-8, np.sin(-8)/-8))
        
        limit_text = Text("Limit = 1", font_size=24, color=WHITE_COLOR)
        self.place_at_grid(limit_text, "A4")

        self.play(Create(axes_final), Create(graph_final))
        self.play(FadeIn(limit_dot))
        
        # Move dot along the path to the limit
        self.play(
            MoveAlongPath(limit_dot, graph_final),
            run_time=3,
            rate_func=linear
        )
        
        # Pulse at the limit point (0,1)
        target_point = axes_final.c2p(0, 1)
        self.play(limit_dot.animate.move_to(target_point))
        self.play(Write(limit_text))
        
        # Visual 'black hole' / hole at (0,1)
        hole = Circle(radius=0.1, color=WHITE_COLOR, fill_color=BLACK, fill_opacity=1)
        hole.move_to(target_point)
        self.play(FadeIn(hole))
        
        self.wait(2)
