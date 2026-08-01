from manim import *
import numpy as np

CYAN = "#00FFFF"

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

class Section5ApplicationScene(TeachingScene):
    def construct(self):
        # Setup the scene with title and lecture lines
        title_text = "Real-World Application: The Robotic Laser"
        lecture_lines = [
            "Robo-A tracks an elliptical path for laser cutting.",
            "We need instantaneous direction at any coordinate.",
            "Implicit differentiation calculates the laser's path instantly."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Task: Draw an ellipse with equation 4x^2 + 9y^2 = 36 in white (#FFFFFF)
        ellipse = Ellipse(width=3.0, height=2.0, color=WHITE)
        self.place_at_grid(ellipse, "C4")
        
        # FIX: Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        ellipse_label = Text("4x^2 + 9y^2 = 36", color=WHITE)
        self.place_at_grid(ellipse_label, "A4", scale_factor=0.8)
        
        self.play(
            Create(ellipse),
            Write(ellipse_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Task: Mark a point at (0, 2) with a yellow dot (#FFFF00) representing the laser head
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        laser_dot = Dot(color=YELLOW)
        self.place_at_grid(laser_dot, "B4")
        
        # Label for the point at grid B3 (to the left of the peak)
        dot_label = Text("(0, 2)", color=YELLOW)
        self.place_at_grid(dot_label, "B3", scale_factor=0.6)
        
        self.play(
            FadeIn(laser_dot),
            Write(dot_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Task: Draw a horizontal line segment tangent to the ellipse at (0, 2) in cyan (#00FFFF)
        self.play(self.lecture[2].animate.set_color(CYAN))
        
        # Tangent is horizontal at the peak. Create a line segment centered at B4.
        tangent_line = Line(start=LEFT * 0.6, end=RIGHT * 0.6, color=CYAN)
        self.place_at_grid(tangent_line, "B4")
        
        # Derivative label at grid B5 (to the right of the peak)
        slope_label = Text("dy/dx = 0", color=CYAN)
        self.place_at_grid(slope_label, "B5", scale_factor=0.7)
        
        self.play(
            Create(tangent_line),
            Write(slope_label)
        )
        self.wait(2)
