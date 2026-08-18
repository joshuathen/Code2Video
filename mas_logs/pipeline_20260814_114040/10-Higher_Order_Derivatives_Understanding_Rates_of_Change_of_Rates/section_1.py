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
        lecture_lines = [
            "The derivative f'(x) represents the instantaneous rate of change.",
            "Visualize f(x) as position.",
            "Then f'(x) represents velocity."
        ]
        self.setup_layout("Prerequisite Review: The Velocity of Motion", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        formula = MathTex("f'(x)", color="#FFFFFF")
        self.place_at_grid(formula, "B5", scale_factor=1.5)
        self.play(Write(formula))
        
        # Adding asset-tagged dummy (none.svg does not exist, using basic shapes)
        path = Line(self.grid["E1"], self.grid["E6"], color="#FFFFFF")
        self.place_at_grid(path, "E2", scale_factor=0.5)
        dot = Dot(color="#FFFFFF")
        dot.move_to(path.get_start())
        self.play(dot.animate.move_to(path.get_end()), run_time=2)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00")
        axes = Axes(x_range=[0, 5], y_range=[0, 2], axis_config={"color": "#FFFF00"}).scale(0.5)
        self.place_in_area(axes, "B3", "E6", scale_factor=0.6)
        line = axes.plot(lambda x: 1, color="#FFFF00")
        self.play(Create(axes), Create(line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        accel_line = axes.plot(lambda x: x*0.3 + 0.5, color="#FF00FF")
        self.play(ReplacementTransform(line, accel_line))
        self.wait(1)
