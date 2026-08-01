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
        # Setup layout with title and lines
        lines = [
            'One equation can have many possible solutions.',
            'An initial condition picks one unique path.',
            'This specific path is the particular solution.'
        ]
        self.setup_layout("General vs. Particular Solutions", lines)

        # Colors
        COLOR_GENERAL = "#FFFFFF"
        COLOR_POINT = "#FF0000"
        COLOR_PARTICULAR = "#FFD700"

        # Create Axes
        axes = Axes(
            x_range=[-2, 2, 1],
            y_range=[-2, 5, 1],
            x_length=5,
            y_length=5,
            axis_config={"color": GREY_C},
            tips=False
        )
        self.place_in_area(axes, "A1", "F6")

        # Family of curves: y = e^x + C
        # We'll use C values: -1.5, -0.5, 0.5, 1.0, 2.0
        # The point (0, 2) corresponds to 2 = e^0 + C => C = 1
        c_values = [-1.5, -0.5, 0.5, 1.0, 2.0]
        curves = VGroup()
        particular_curve_index = 3 # index of C=1.0

        for c in c_values:
            curve = axes.plot(lambda x: np.exp(x) + c, x_range=[-2, 1.4], color=COLOR_GENERAL)
            curves.add(curve)

        # Label for General Solution - Replaced MathTex with Text
        general_label = Text("y = e^x + C", color=COLOR_GENERAL, font_size=32)
        # Resolved Issue #50: Positioned in area for better balance
        self.place_in_area(general_label, 'A3', 'B4', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_GENERAL))
        self.play(Create(axes), run_time=1)
        self.play(Create(curves), Write(general_label), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight initial condition (0, 2)
        dot = Dot(axes.coords_to_point(0, 2), color=COLOR_POINT)
        # Replaced MathTex with Text
        dot_label = Text("(0, 2)", color=COLOR_POINT, font_size=24)
        dot_label.next_to(dot, UR, buff=0.1)

        self.play(self.lecture[1].animate.set_color(COLOR_POINT))
        self.play(FadeIn(dot), Write(dot_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fade out non-particular curves and change the chosen one to GOLD
        particular_curve = curves[particular_curve_index]
        other_curves = VGroup(*[curves[i] for i in range(len(curves)) if i != particular_curve_index])
        
        particular_solution_label = Text("Particular Solution", color=COLOR_PARTICULAR, font_size=24)
        # Resolved Issue #48: Positioned in area to avoid x-axis overlap
        self.place_in_area(particular_solution_label, 'E2', 'E4', scale_factor=0.7)
        
        # Replaced MathTex with Text
        particular_formula = Text("y = e^x + 1", color=COLOR_PARTICULAR, font_size=36)
        particular_formula.set_stroke(width=1) # Visual bolding
        # Resolved Issue #49: Positioned in area to avoid edge cramping
        self.place_in_area(particular_formula, 'F2', 'F4', scale_factor=0.8)

        self.play(self.lecture[2].animate.set_color(COLOR_PARTICULAR))
        self.play(
            FadeOut(other_curves),
            FadeOut(general_label),
            particular_curve.animate.set_color(COLOR_PARTICULAR).set_stroke(width=6),
            FadeIn(particular_solution_label),
            FadeIn(particular_formula)
        )
        self.wait(2)
