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
        self.setup_layout("Intuitive Hook: The Perspective Shift", [
            "Coordinates describe space like a language.",
            "Robby moves across the grid floor.",
            "Standard grid vs. tilted grid systems."
        ])

        # Animation Elements
        grid = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_numbers": False})
        
        # Grid group for better positioning
        grid_group = VGroup(grid)
        self.place_in_area(grid_group, 'B2', 'E4', scale_factor=0.85)
        
        vector = Vector([1, 1], color="#00FFFF")
        vector.shift(grid.c2p(0, 0) - vector.get_start())
        
        # Asset: Robot icon
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        robot.scale(0.3)
        robot.next_to(vector.get_end(), UP, buff=0.1)
        
        # Label
        vector_v_label = MathTex("v", color="#00FFFF")
        self.place_at_grid(vector_v_label, 'D5', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(grid), run_time=1)
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(GrowArrow(vector), FadeIn(robot), run_time=1)
        self.lecture[1].set_color("#FFFFFF")
        self.add(vector_v_label)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        origin_dot = Dot(grid.c2p(0, 0), color="#FFD700")
        self.play(FadeIn(origin_dot))
        
        self.play(Rotate(grid, angle=PI/4, about_point=grid.c2p(0, 0)), 
                  Rotate(robot, angle=PI/4, about_point=grid.c2p(0, 0)),
                  run_time=2)
        self.wait(2)
