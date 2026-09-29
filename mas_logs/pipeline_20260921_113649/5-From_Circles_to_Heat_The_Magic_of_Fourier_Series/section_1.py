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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisites: The Geometry of Cycles", [
            "Euler's formula describes rotation via complex numbers.",
            "We represent this as a phasor tracing a circle.",
            "Rotation in the complex plane creates a cycle."
        ])
        
        # Complex plane coordinate system
        axes = Axes(
            x_range=[-1.5, 1.5, 1],
            y_range=[-1.5, 1.5, 1],
            axis_config={"include_numbers": False, "include_tip": True}
        )
        self.place_in_area(axes, 'A3', 'F6', scale_factor=0.6)
        self.add(axes)
        
        # Asset compass
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(compass, "B4", scale_factor=0.3)
        self.add(compass)
        
        circle = Circle(radius=1, color="#FFFFFF")
        self.place_at_grid(circle, "B4", scale_factor=0.6)
        
        point = Dot(color="#FF00FF")
        phasor = Line(start=circle.get_center(), end=circle.get_start(), color="#00FFFF")
        
        # Trackers
        angle = ValueTracker(0)
        phasor.add_updater(lambda m: m.put_start_and_end_on(
            circle.get_center(),
            circle.get_center() + np.array([np.cos(angle.get_value()), np.sin(angle.get_value()), 0]) * circle.radius
        ))
        point.add_updater(lambda m: m.move_to(phasor.get_end()))
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(circle), FadeIn(compass), run_time=1)
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        self.add(phasor, point)
        self.play(angle.animate.set_value(2 * PI), run_time=3, rate_func=linear)
        self.lecture[1].set_color("#FF00FF")
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.wait(1)
