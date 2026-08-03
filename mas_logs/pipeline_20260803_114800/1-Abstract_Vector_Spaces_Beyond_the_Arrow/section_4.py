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
        self.setup_layout("Expanding the Mind: Polynomials as Vectors", [
            "Polynomials can also behave exactly like vectors.",
            "Adding two quadratic curves results in another curve.",
            "Scaling a polynomial keeps it within the same family.",
            "Their coefficients act just like coordinates in space.",
            "Thus, polynomials form a valid abstract vector space."
        ])

        # Colors
        ORANGE_RED = "#FF4500"
        DODGER_BLUE = "#1E90FF"
        GREEN_YELLOW = "#ADFF2F"
        WHITE_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(ORANGE_RED)
        
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-1, 4, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": False, "include_numbers": False}
        )
        curve1 = axes.plot(lambda x: x**2, x_range=[-2, 2], color=ORANGE_RED)
        curve1_label = MathTex("y = x^2", color=ORANGE_RED, font_size=32)
        
        plot_group = VGroup(axes, curve1)
        self.place_in_area(plot_group, "A1", "D6", scale_factor=0.8)
        
        # Place label near curve
        curve1_label.next_to(curve1, UP, buff=0.1)
        
        self.play(Write(axes), Create(curve1), run_time=1.5)
        self.play(Write(curve1_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(DODGER_BLUE)
        
        line_curve = axes.plot(lambda x: x, x_range=[-2, 2], color=DODGER_BLUE)
        line_label = MathTex("y = x", color=DODGER_BLUE, font_size=32)
        line_label.next_to(line_curve, DOWN, buff=0.1)
        
        sum_curve = axes.plot(lambda x: x**2 + x, x_range=[-2, 2], color=GREEN_YELLOW)
        sum_label = MathTex("y = x^2 + x", color=GREEN_YELLOW, font_size=32)
        # Fix for Issue 27: Move from D6 to E6 to avoid overlap with axes
        self.place_at_grid(sum_label, "E6", scale_factor=0.8)

        self.play(Create(line_curve), Write(line_label), run_time=1.5)
        self.wait(1)
        
        self.play(
            FadeOut(curve1_label),
            FadeOut(line_label),
            Transform(curve1, sum_curve),
            Transform(line_curve, sum_curve),
            run_time=2
        )
        self.remove(line_curve)
        active_curve = curve1 # curve1 is now visually sum_curve
        self.play(Write(sum_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN_YELLOW)
        
        scaled_curve = axes.plot(lambda x: 2*(x**2 + x), x_range=[-1.5, 1.2], color=GREEN_YELLOW)
        scaled_label = MathTex("y = 2(x^2 + x)", color=GREEN_YELLOW, font_size=32)
        # Fix for Issue 27: Move from D6 to E6
        self.place_at_grid(scaled_label, "E6", scale_factor=0.8)

        self.play(
            Transform(active_curve, scaled_curve),
            Transform(sum_label, scaled_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(WHITE_COLOR)
        
        coords = MathTex("[1, 1, 0]", color=WHITE_COLOR, font_size=40)
        # Fix for Issue 28: Center in E3-E4 area
        self.place_in_area(coords, "E3", "E4", scale_factor=1.0)
        
        self.play(Write(coords))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(WHITE_COLOR)
        
        p2_space = MathTex("P_2 \\text{ Space}", color=WHITE_COLOR, font_size=48)
        # Fix for Issue 29: Center in F3-F4 area
        self.place_in_area(p2_space, "F3", "F4", scale_factor=1.0)
        
        self.play(
            Write(p2_space),
            p2_space.animate.set_color(YELLOW).set_color(WHITE_COLOR),
            Indicate(p2_space, color=WHITE_COLOR),
            run_time=2
        )
        self.wait(2)
