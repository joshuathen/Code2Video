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
        lecture_lines = [
            "Position graph shows location over time.",
            "Velocity graph shows slope of position.",
            "Acceleration graph shows slope of velocity.",
            "Steep velocity indicates high acceleration.",
            "Flat velocity means zero acceleration."
        ]
        self.setup_layout("Visualizing Physical Meaning", lecture_lines)
        
        # Setup axes
        axes_config = {"x_range": [0, 4, 1], "y_range": [0, 4, 1], "x_length": 3, "y_length": 2, "tips": False}
        
        # Use SVGMobject for the object icon
        dot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/object.svg", color=WHITE)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF0000")
        pos_axes = Axes(**axes_config)
        pos_curve = pos_axes.plot(lambda x: 0.25 * x**2, color="#FF0000")
        pos_group = VGroup(pos_axes, pos_curve)
        self.place_in_area(pos_group, 'A3', 'C6', scale_factor=0.8)
        self.add(pos_group)
        
        # Place dot on position graph
        self.place_at_grid(dot, 'B4', scale_factor=0.2)
        self.add(dot)
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        vel_axes = Axes(**axes_config)
        vel_curve = vel_axes.plot(lambda x: 0.5 * x, color="#00FF00")
        vel_group = VGroup(vel_axes, vel_curve)
        self.place_at_grid(vel_group, 'D2', scale_factor=0.9)
        self.add(vel_group)
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#0000FF")
        acc_axes = Axes(**axes_config)
        acc_curve = acc_axes.plot(lambda x: 0.5, color="#0000FF")
        acc_group = VGroup(acc_axes, acc_curve)
        self.place_at_grid(acc_group, 'D5', scale_factor=0.9)
        self.add(acc_group)
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFFFF")
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#888888")
        self.wait(2)
