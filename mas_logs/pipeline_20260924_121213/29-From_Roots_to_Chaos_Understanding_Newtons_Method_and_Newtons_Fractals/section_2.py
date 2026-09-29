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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Newton-Raphson Iteration", [
            "Use formula: x_{n+1} = x_n - f(x_n)/f'(x_n).",
            "It creates a sawtooth path.",
            "Robotic arms find equilibrium points."
        ])
        
        # Load asset
        robot_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        
        # Formula
        formula = MathTex("x_{n+1} = x_n - \\frac{f(x_n)}{f'(x_n)}", color="#FFFF00")
        self.place_at_grid(formula, 'B2', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        # Using asset per issue 17
        robot_ref1 = robot_svg.copy()
        self.place_at_grid(robot_ref1, 'A5', scale_factor=0.3)
        self.play(FadeIn(formula), FadeIn(robot_ref1))
        self.lecture[0].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Create a simplified function and sawtooth visualization
        axes = Axes(x_range=[-1, 3], y_range=[-1, 3], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: (x-1)**2, color="#FF0000")
        
        # Visual objects
        path = VGroup()
        x = 2.5
        for i in range(3):
            y = axes.c2p(x, (x-1)**2)
            pt = Dot(y, color="#FF0000")
            path.add(pt)
            # Simple jump
            x = x - ((x-1)**2) / (2*(x-1))
            
        self.place_in_area(axes, 'C1', 'F3', scale_factor=0.5)
        self.play(Create(axes), Create(curve))
        self.play(Create(path))
        self.lecture[1].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        robot_ref2 = robot_svg.copy()
        self.place_at_grid(robot_ref2, 'C5', scale_factor=0.7)
        self.play(FadeIn(robot_ref2))
        self.lecture[2].set_color("#00FF00")
        self.wait(1)
