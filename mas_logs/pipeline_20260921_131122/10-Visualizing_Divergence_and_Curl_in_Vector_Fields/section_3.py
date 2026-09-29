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
        self.setup_layout("Curl: The Micro-Rotation", [
            "Curl measures rotation at a point.", 
            "A paddlewheel detects local spinning.", 
            "Counter-clockwise spin means positive curl."
        ])
        
        # Create elements
        paddle = VGroup(
            Line(UP*0.5, DOWN*0.5),
            Line(LEFT*0.5, RIGHT*0.5),
            Circle(radius=0.5, color=BLUE)
        )
        
        # Apply positioning per feedback
        # Using self.place_in_area as requested in issue 27/42
        self.place_in_area(paddle, 'B4', 'E6', scale_factor=1.2)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(paddle))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        paddle_rotation = Rotating(paddle, radians=PI/2, about_point=paddle.get_center(), run_time=2)
        self.play(paddle_rotation)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        arrow = Arrow(start=UP*0.8, end=UP*1.5, color=RED)
        # Apply positioning per feedback
        self.place_at_grid(arrow, 'B5', scale_factor=0.6)
        self.play(GrowArrow(arrow))
        self.wait(1)
