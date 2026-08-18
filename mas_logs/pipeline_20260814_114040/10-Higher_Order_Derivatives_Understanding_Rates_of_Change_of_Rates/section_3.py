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
            "If f''(x) > 0, the curve is concave up.",
            "If f''(x) < 0, the curve is concave down.",
            "Concavity visually describes the curve's bend."
        ]
        self.setup_layout("Visualizing Concavity", lecture_lines)
        
        # Adjust axes as per feedback (Issue 25)
        axes = Axes(x_range=[-2, 2], y_range=[-1, 3], axis_config={"include_tip": False}).scale(0.5)
        self.place_in_area(axes, "C2", "F5", scale_factor=0.6)
        self.add(axes)
        
        # === Animation for Lecture Line 1 ===
        # Using cup asset (Issue 18)
        cup = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cup.svg", color="#00FF00")
        self.place_at_grid(cup, "B4", scale_factor=0.3)
        curve_up = axes.plot(lambda x: x**2, color="#00FF00")
        self.play(Create(curve_up), FadeIn(cup))
        self.lecture[0].set_color("#00FF00") # Issue 26
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Using bowl asset (Issue 18)
        bowl = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bowl.svg", color="#40E0D0")
        self.place_at_grid(bowl, "B5", scale_factor=0.3)
        curve_down = axes.plot(lambda x: -x**2 + 2, color="#40E0D0")
        self.play(Create(curve_down), FadeIn(bowl))
        self.lecture[1].set_color("#40E0D0") # Issue 27
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        inflection_dot = Dot(color="#FFFFFF").move_to(axes.c2p(0, 1))
        self.play(FadeIn(inflection_dot), run_time=1)
        self.lecture[2].set_color("#FFFFFF")
        self.wait(2)
