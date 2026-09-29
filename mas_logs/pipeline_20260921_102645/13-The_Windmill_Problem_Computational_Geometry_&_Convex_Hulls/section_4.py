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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Core Insight: Periodicity and Convex Hull", [
            "The line visits every point pair.",
            "Rotation is periodic and cyclic.",
            "It reveals the convex hull.",
            "Matches the rotating calipers algorithm.",
            "An elegant geometric structure emerges."
        ])

        # Points setup
        points = [
            np.array([0, 1, 0]), np.array([1, 1.5, 0]), np.array([2, 0.5, 0]),
            np.array([1.5, -1, 0]), np.array([0.5, -0.5, 0]), np.array([-0.5, 0, 0])
        ]
        dots = VGroup(*[Dot(p, color=BLUE) for p in points])
        self.place_in_area(dots, "A2", "F6", scale_factor=1.2)
        
        hull = Polygon(dots[0].get_center(), dots[1].get_center(), dots[2].get_center(), 
                       dots[3].get_center(), dots[4].get_center(), dots[5].get_center(), color=YELLOW)
        
        # Windmill asset
        windmill = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/windmill.svg")
        self.place_at_grid(windmill, "C3", scale_factor=0.5)

        line = Line(start=np.array([-2, 0, 0]), end=np.array([2, 0, 0]), color=WHITE)
        line.move_to(dots[0].get_center())

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"), Create(dots))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"), Create(hull), FadeIn(windmill))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"), Create(line))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FFFF"), line.animate.rotate(PI/4, about_point=dots[0].get_center()))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF4500"), line.animate.set_color("#FF4500"))
