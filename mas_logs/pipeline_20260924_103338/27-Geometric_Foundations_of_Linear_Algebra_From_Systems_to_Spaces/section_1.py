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
            "Linear equations define constraints in our spatial world.",
            "Imagine a robot arm constrained by these equations.",
            "The intersection of these constraints identifies a pose."
        ]
        self.setup_layout("Geometric Interpretation of Linear Systems", lecture_lines)
        
        # Axis and setup
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, "B3", "E6", scale_factor=0.7)
        self.add(axes)
        
        # Lines 1: Linear constraints
        line1 = Line(axes.c2p(-3, 1), axes.c2p(3, -1), color="#FF5733")
        line2 = Line(axes.c2p(-1, -3), axes.c2p(1, 3), color="#FF5733")
        intersection = Dot(axes.c2p(0, 0), color="#FFFF33")
        
        # Asset for robot arm
        robot_arm = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        self.play(Create(line1), Create(line2))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#33FF57")
        self.place_at_grid(robot_arm, "D4", scale_factor=0.3)
        self.play(FadeIn(robot_arm))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF33")
        # Apply constraint from issue 27/42
        equation = MathTex(r"A x = b").set_color("#FFFF33")
        self.place_at_grid(equation, "D2", scale_factor=0.7)
        self.play(Write(equation), FadeIn(intersection))
        
        self.wait(2)
