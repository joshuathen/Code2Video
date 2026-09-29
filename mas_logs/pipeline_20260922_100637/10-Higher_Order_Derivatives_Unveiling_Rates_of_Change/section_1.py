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
        self.setup_layout("Prerequisite Review: The Velocity of Change", [
            "The first derivative measures instantaneous rate of change.",
            "Think of a speedometer showing velocity at time t.",
            "It describes how position changes over time."
        ])
        
        # Elements
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 4, 1], axis_config={"include_numbers": False}).scale(0.4)
        self.place_in_area(axes, 'B3', 'E6', scale_factor=0.6)
        curve = axes.plot(lambda x: 0.2*x**3, color=WHITE)
        
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        self.place_at_grid(speedometer, 'B6', scale_factor=0.25)
        
        label_f_prime = MathTex("f'(x)", color="#FFFF00")
        self.place_at_grid(label_f_prime, 'E5', scale_factor=0.6)
        
        tangent = Line(start=LEFT, end=RIGHT, color="#FF00FF").scale(0.3)
        dot = Dot(color="#FFFF00")
        
        # Manual updater logic
        tracker = ValueTracker(0)
        def update_tangent(t):
            x = tracker.get_value()
            y = 0.2 * x**3
            t.move_to(axes.c2p(x, y))
            slope = 0.6 * x**2
            t.set_angle(np.arctan(slope))
        
        tangent.add_updater(update_tangent)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(axes), Create(curve), FadeIn(speedometer))
        self.lecture[0].set_color("#FFFFFF")
        
        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(label_f_prime), FadeIn(tangent), FadeIn(dot))
        self.lecture[1].set_color("#FFFF00")
        
        # === Animation for Lecture Line 3 ===
        self.play(tracker.animate.set_value(3), run_time=4)
        self.lecture[2].set_color("#FF00FF")
        
        self.wait(2)
        tangent.remove_updater(update_tangent)
