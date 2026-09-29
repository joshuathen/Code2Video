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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Summary and Implications", [
            "Newton's method finds function roots.",
            "Sensitive starts yield chaotic fractals.",
            "Simple rules create infinite complexity."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Summarize the importance of the Newton method. Color: #FFFF00.
        self.lecture[0].set_color("#FFFF00")
        
        root_dot = Dot(color="#FFFF00")
        self.place_at_grid(root_dot, 'B5', scale_factor=0.8)
        self.play(FadeIn(root_dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show the link between geometry and chaos. Color: #00FFFF.
        self.lecture[1].set_color("#00FFFF")
        
        dividing_line = Line(start=self.grid["C1"], end=self.grid["C6"], color="#00FFFF")
        self.place_in_area(dividing_line, 'C1', 'C6', scale_factor=0.9)
        self.play(Create(dividing_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Display concluding visual of a beautiful fractal. Color: #FFFFFF.
        self.lecture[2].set_color("#FFFFFF")
        
        circle_group = VGroup(*[Circle(radius=0.5, color="#FFFFFF") for _ in range(3)])
        circle_group.arrange(RIGHT, buff=0.2)
        self.place_in_area(circle_group, 'D2', 'F5', scale_factor=0.7)
        self.play(FadeIn(circle_group))
        self.wait(2)
