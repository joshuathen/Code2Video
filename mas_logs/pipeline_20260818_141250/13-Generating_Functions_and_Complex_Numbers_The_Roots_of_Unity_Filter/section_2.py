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
        self.setup_layout("Prerequisite: The Geometry of Complex Numbers", [
            "Euler’s formula: e^(iθ) equals cosθ + i sinθ.",
            "Roots of unity are evenly spaced on the circle.",
            "Their sum over a cycle always vanishes."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        
        circle = Circle(radius=1.5, color=WHITE)
        self.place_in_area(circle, "B2", "E5")
        
        euler_label = MathTex(r"e^{i\theta} = \cos\theta + i \sin\theta", color="#FFFFFF")
        euler_label.next_to(circle, UP)
        
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg", color=WHITE)
        self.place_at_grid(compass, "B5", scale_factor=0.3)
        
        self.play(Create(circle), Write(euler_label), FadeIn(compass))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        
        roots = [np.exp(2j * PI * k / 3) for k in range(3)]
        points = VGroup(*[Dot(circle.point_from_proportion((k / 3) % 1), color="#00FFFF") for k in range(3)])
        self.play(Create(points))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF6347")
        
        vectors = VGroup(*[
            Arrow(start=ORIGIN, end=points[k].get_center(), buff=0, color="#FF6347")
            for k in range(3)
        ])
        
        self.play(Create(vectors))
        self.wait(2)
