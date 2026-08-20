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
        self.setup_layout("The Mathematical Mapping: Phase Space", [
            "Map collisions to geometry in phase space.",
            "Energy conservation defines an ellipse.",
            "Collisions correspond to reflections off the ellipse."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Axes setup
        axes = Axes(
            x_range=[-1, 5], y_range=[-1, 5],
            x_length=4, y_length=4,
            axis_config={"include_numbers": False}
        )
        self.place_in_area(axes, 'B3', 'F6', scale_factor=0.9)
        self.play(Create(axes))
        self.lecture[0].set_color("#00FF00")

        # === Animation for Lecture Line 2 ===
        # Ellipse: Energy Conservation x^2/a^2 + y^2/b^2 = 1
        ellipse = Ellipse(width=3, height=2, color="#FF00FF")
        self.place_at_grid(ellipse, 'D4', scale_factor=1.2)
        
        # Particle asset
        particle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/particle.svg")
        self.place_at_grid(particle, 'C5', scale_factor=0.5)

        self.play(FadeIn(ellipse), FadeIn(particle))
        self.lecture[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        # Trajectory segment as reflection
        trajectory = Arc(radius=1.5, start_angle=PI/2, angle=-PI/4, color="#FFFF00")
        self.place_at_grid(trajectory, 'D4', scale_factor=1.1)
        self.play(Create(trajectory))
        self.lecture[2].set_color("#FFFF00")
