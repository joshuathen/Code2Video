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
        self.setup_layout("Graphical Interpretation and Application", [
            "Derivative equals slope of tangent line.",
            "It measures instantaneous rate of change.",
            "Think of your car's speedometer reading.",
            "Calculus bridges algebra and geometry.",
            "Instantaneous speed means speed right now."
        ])

        # Setup Axes and Curve
        axes = Axes(x_range=[-1, 3], y_range=[-1, 5], axis_config={"include_tip": False}).scale(0.4)
        curve = axes.plot(lambda x: x**2, color=BLUE)
        graph_group = VGroup(axes, curve)
        self.place_in_area(graph_group, 'B4', 'E6', scale_factor=0.5)

        # Assets
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        tangent = TangentLine(curve, alpha=0.5, length=2, color=YELLOW)
        self.add(tangent)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        dot = Dot(color=RED)
        self.place_at_grid(dot, 'B3', scale_factor=0.6)
        self.add(dot)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        speedometer_label = Text("Speedometer", font_size=20, color=GREEN)
        self.place_at_grid(speedometer_label, 'C3', scale_factor=0.7)
        self.add(speedometer_label)
        self.place_at_grid(speedometer, 'D3', scale_factor=0.8)
        self.add(speedometer)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(PURPLE)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(ORANGE)
        self.play(Indicate(dot))
