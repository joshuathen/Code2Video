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
            "Curl measures a field's tendency to rotate.",
            "Cross product models the axis of rotation.",
            "Paddlewheel toys visualize local rotational curl."
        ]
        self.setup_layout("Curl: The 'Rotation' Concept", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Static vector field representing rotation (using Arrow)
        rot_field = VGroup()
        for i in range(-2, 3):
            for j in range(-2, 3):
                vec = Arrow(start=ORIGIN, end=0.3*np.array([-j, i, 0]), buff=0, color=YELLOW)
                rot_field.add(vec)
        self.place_in_area(rot_field, 'A4', 'B6', scale_factor=0.6)
        self.play(FadeIn(rot_field))
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # === Animation for Lecture Line 2 ===
        # Highlight the curl operator circulating around a point
        circle = Circle(radius=0.8, color=BLUE, stroke_width=4)
        center_point = Dot(color=BLUE)
        self.place_at_grid(circle, 'D3', scale_factor=0.7)
        self.place_at_grid(center_point, 'D3', scale_factor=0.7)
        curl_op = Tex(r"$\nabla \times \mathbf{F}$", color=BLUE).next_to(circle, DOWN)
        
        self.play(Create(circle), FadeIn(center_point), Write(curl_op))
        self.play(self.lecture[1].animate.set_color(BLUE))

        # === Animation for Lecture Line 3 ===
        # Paddlewheel toy visualize local rotational curl
        paddle = VGroup(
            Line(UP*0.5, DOWN*0.5, color="#00FFFF"),
            Line(LEFT*0.5, RIGHT*0.5, color="#00FFFF")
        )
        self.place_at_grid(paddle, 'D5', scale_factor=0.7)
        self.play(Rotate(paddle, angle=PI/2, run_time=2))
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.wait(2)
