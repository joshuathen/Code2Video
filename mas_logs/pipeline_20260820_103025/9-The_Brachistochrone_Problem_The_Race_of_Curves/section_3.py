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
        self.setup_layout("The Geometry of the Solution: The Cycloid", [
            "The Brachistochrone solution is a cycloid.",
            "Trace a point on a rolling circle.",
            "It balances speed gain and distance reduction.",
            "The cycloid curve is mathematically optimal.",
            "Geometry holds the key to the fastest time."
        ])
        
        # Setup Cycloid animation components
        radius = 0.4
        wheel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color=BLUE)
        point = Dot(color=RED)
        line = Line(start=[-1, 0, 0], end=[3, 0, 0], color=WHITE)
        
        # Position using fixes from Critic
        self.place_at_grid(line, 'C2', scale_factor=1.2)
        
        # Group circle and point
        circle_curve_group = VGroup(wheel, point)
        self.place_in_area(circle_curve_group, 'C4', 'E6', scale_factor=0.8)
        
        # Rolling circle label
        rolling_circle_label = Text("Rolling circle", font_size=20, color=BLUE)
        self.place_at_grid(rolling_circle_label, 'C5', scale_factor=0.7)
        rolling_circle_label.next_to(wheel, UP, buff=0.1)

        wheel_tracker = ValueTracker(0)
        
        # Wheel updater
        wheel.add_updater(lambda m: m.move_to(
            line.get_start() + np.array([wheel_tracker.get_value(), radius, 0])
        ))
        
        # Point updater
        point.add_updater(lambda m: m.move_to(
            wheel.get_center() + np.array([
                -radius * np.sin(wheel_tracker.get_value() / radius),
                -radius * np.cos(wheel_tracker.get_value() / radius),
                0
            ])
        ))
        
        cycloid = TracedPath(point.get_center, stroke_color=YELLOW, stroke_width=4)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(RED)
        self.add(wheel, point, cycloid, rolling_circle_label)
        self.play(wheel_tracker.animate.set_value(2 * PI * radius), run_time=3, rate_func=linear)
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(GREEN)
        self.wait(2)
        
        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(Indicate(cycloid, color=YELLOW), run_time=1.5)
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(BLUE)
        self.wait(2)
