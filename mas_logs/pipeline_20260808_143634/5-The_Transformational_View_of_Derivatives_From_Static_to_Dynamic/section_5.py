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
        self.setup_layout("Summary and Synthesis", [
            "Derivative is instantaneous rate of change.",
            "It is the slope of a tangent.",
            "Curves are series of linear segments."
        ])
        
        # Animation Elements
        curve = FunctionGraph(lambda x: 0.5 * np.sin(3*x) + 0.5 * x**2, x_range=[-1, 2], color=BLUE)
        self.place_in_area(curve, 'B3', 'E6', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(curve))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        tangent_line = Line(start=LEFT*1, end=RIGHT*1, color=RED).rotate(0.5)
        self.place_at_grid(tangent_line, 'C4', scale_factor=0.7)
        self.play(FadeIn(tangent_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(ORANGE))
        # Create series of segments
        segments = VGroup()
        for i in range(10):
            x = -0.8 + i * 0.3
            y = 0.5 * np.sin(3*x) + 0.5 * x**2
            seg = Line(start=np.array([x, y, 0]), end=np.array([x+0.2, y+0.1, 0]), color=WHITE)
            segments.add(seg)
        self.place_in_area(segments, 'D2', 'F5', scale_factor=0.6)
        self.play(FadeOut(curve), FadeOut(tangent_line))
        self.play(Create(segments))
        self.wait(2)
