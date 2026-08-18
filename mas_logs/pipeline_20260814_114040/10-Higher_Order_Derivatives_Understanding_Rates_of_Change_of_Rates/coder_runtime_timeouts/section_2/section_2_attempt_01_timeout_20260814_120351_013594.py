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
        lecture_lines = [
            "The second derivative f''(x) is the derivative of f'(x).",
            "It measures acceleration, or how velocity changes.",
            "Graphically, it determines the curve's concavity."
        ]
        self.setup_layout("Defining the Second Derivative", lecture_lines)
        
        # Setup Axes
        axes = Axes(x_range=[-2, 2], y_range=[-1, 3], x_length=4, y_length=3).shift(self.grid["C3"])
        f_prime = axes.plot(lambda x: x**2 + 0.5, color="#00FFFF")
        self.add(axes, f_prime)

        # === Animation for Lecture Line 1 ===
        # The second derivative f''(x) is the derivative of f'(x).
        self.lecture[0].set_color("#00FFFF")
        
        # === Animation for Lecture Line 2 ===
        # It measures acceleration, or how velocity changes.
        self.lecture[1].set_color("#FF9900")
        
        tracker = ValueTracker(-1.5)
        tangent = always_redraw(lambda: TangentLine(f_prime, alpha=axes.c2p(tracker.get_value(), 0)[0], length=1, color="#FF9900"))
        self.add(tangent)
        self.play(tracker.animate.set_value(1.5), run_time=3)
        
        # === Animation for Lecture Line 3 ===
        # Graphically, it determines the curve's concavity.
        self.lecture[2].set_color("#FF0000")
        
        slope_val = DecimalNumber(0, num_decimal_places=2, color="#FF0000")
        self.place_at_grid(slope_val, "E3")
        slope_val.add_updater(lambda d: d.set_value(2 * tracker.get_value()))
        self.add(slope_val)
        
        self.play(tracker.animate.set_value(-1.5), run_time=2)
