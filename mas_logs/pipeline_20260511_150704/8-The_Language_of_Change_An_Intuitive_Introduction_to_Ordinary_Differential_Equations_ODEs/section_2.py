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
        # Setup layout with specific title and lecture lines
        self.setup_layout(
            "Prerequisite Bridge: The Slope as a Rate", 
            [
                'The derivative represents the slope at a point.', 
                'It tells us our instantaneous rate of change.', 
                'Knowing every slope allows us to reconstruct paths.'
            ]
        )
        
        # Color constants from instruction
        COLOR_CURVE = "#888888"
        COLOR_TANGENT = "#FF00FF"
        COLOR_VALUE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_CURVE))
        
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-1, 9, 1],
            x_length=5,
            y_length=4,
            axis_config={"include_tip": True, "color": WHITE}
        )
        # Position axes in the designated area
        self.place_in_area(axes, "B3", "F6", scale_factor=0.7)
        
        # Drawing a "hill" curve: y = 5 - 0.5x^2
        curve = axes.plot(lambda x: 5 - 0.5 * (x**2), color=COLOR_CURVE)
        
        # === Animation for Lecture Line 2 ===
        point_tracker = ValueTracker(1.5)
        
        # Tangent line and dot updates
        def get_tangent_line():
            val = point_tracker.get_value()
            # Map val in [-3, 3] to alpha in [0, 1] for TangentLine
            alpha = (val + 3) / 6
            return TangentLine(curve, alpha=alpha, length=2.2, color=COLOR_TANGENT)
        
        def get_dot():
            val = point_tracker.get_value()
            return Dot(axes.c2p(val, 5 - 0.5 * (val**2)), color=COLOR_VALUE)

        tangent = always_redraw(get_tangent_line)
        dot = always_redraw(get_dot)

        # Numerical slope indicator (Fixed: Using Text instead of DecimalNumber to avoid LaTeX dependency)
        slope_label = Text("Slope:", font_size=20, color=COLOR_VALUE)
        slope_val = Text("0.00", font_size=20, color=COLOR_VALUE)
        slope_group = VGroup(slope_label, slope_val).arrange(RIGHT, buff=0.1)
        # Position near the top of the axes area
        self.place_at_grid(slope_group, "B4")

        # Slope of 5 - 0.5x^2 is -x. Update text manually to avoid LaTeX.
        slope_val.add_updater(lambda m: m.become(
            Text(f"{-point_tracker.get_value():.2f}", font_size=20, color=COLOR_VALUE)
            .next_to(slope_label, RIGHT, buff=0.1)
        ))

        self.play(Create(axes), Create(curve))
        self.play(Create(dot), Create(tangent), Write(slope_group))
        self.wait(1)
        
        self.play(self.lecture[1].animate.set_color(COLOR_TANGENT))
        # Highlight instantaneous change by moving the point
        self.play(point_tracker.animate.set_value(-2.0), run_time=2.5)
        self.play(point_tracker.animate.set_value(2.0), run_time=2.5)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(COLOR_VALUE))
        self.wait(2)
