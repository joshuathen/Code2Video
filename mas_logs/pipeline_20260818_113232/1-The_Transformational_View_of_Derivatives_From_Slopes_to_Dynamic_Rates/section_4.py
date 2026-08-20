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
        self.setup_layout("Dynamic Application: The Cheetah's Sprint", [
            "Position graphs change over time.", 
            "Derivative yields velocity as slope.", 
            "Instantaneous speed is the derivative.", 
            "Acceleration shows rate of change.", 
            "Motion is captured by derivatives."
        ])
        
        # Colors for lecture lines
        colors = ["#FFD700", "#FF4500", "#00CED1", "#ADFF2F", "#DA70D6"]
        
        # Assets
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        axes = Axes(x_range=[0, 5], y_range=[0, 5], x_length=4, y_length=4)
        curve = axes.plot(lambda x: 0.2 * x**3, color=BLUE)
        axes_and_curve = VGroup(axes, curve)
        self.place_in_area(axes_and_curve, "B3", "F6", scale_factor=0.6)
        self.place_at_grid(cheetah, "A3", scale_factor=0.5)
        self.add(axes_and_curve, cheetah)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(colors[1])
        dot = Dot(color=RED)
        dot.move_to(axes.c2p(0, 0))
        tangent = Line(start=LEFT, end=RIGHT, color=YELLOW).set_length(1)
        self.add(dot, tangent)
        
        # Logic using ValueTracker for animation
        x_tracker = ValueTracker(0)
        def get_pos():
            x = x_tracker.get_value()
            y = 0.2 * x**3
            return axes.c2p(x, y)
        
        dot.add_updater(lambda m: m.move_to(get_pos()))
        tangent.add_updater(lambda m: m.set_angle(np.arctan(0.6 * x_tracker.get_value()**2)).move_to(dot.get_center()))
        
        self.play(x_tracker.animate.set_value(4), run_time=3, rate_func=linear)
        dot.remove_updater(dot.get_updaters()[0])
        tangent.remove_updater(tangent.get_updaters()[0])

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(colors[2])
        speed_label = Text("Speed", font_size=20, color=WHITE)
        self.place_at_grid(speed_label, "F3", scale_factor=1.0)
        self.add(speed_label)
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(colors[3])
        accel_label = Text("Acceleration", font_size=20, color=WHITE)
        self.place_at_grid(accel_label, "F5", scale_factor=1.0)
        self.add(accel_label)
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(colors[4])
        self.wait(2)
