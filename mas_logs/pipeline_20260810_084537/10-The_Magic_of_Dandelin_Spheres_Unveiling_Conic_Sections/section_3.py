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
        self.setup_layout("The Geometric Proof: The Ellipse Case", [
            "Consider the ellipse formed by the slicing plane.",
            "The spheres touch the plane at two foci.",
            "Distances to these points sum to a constant.",
            "This proves the slice is indeed an ellipse."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg]
        ellipse_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=WHITE)
        self.place_in_area(ellipse_icon, 'B3', 'E5', scale_factor=0.75)
        self.play(FadeIn(ellipse_icon))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        focus1 = Dot(color="#00FFFF")
        focus2 = Dot(color="#00FFFF")
        self.place_at_grid(focus1, 'C3', scale_factor=1.2)
        self.place_at_grid(focus2, 'C4', scale_factor=1.2)
        self.play(Flash(focus1), Flash(focus2))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg]
        plane_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg", color="#FF00FF")
        p = Dot(color="#FF00FF")
        self.place_at_grid(p, 'D3', scale_factor=0.9)
        self.place_at_grid(plane_icon, 'B4', scale_factor=0.5)
        line1 = Line(focus1.get_center(), p.get_center(), color="#FF00FF")
        line2 = Line(focus2.get_center(), p.get_center(), color="#FF00FF")
        self.play(Create(line1), Create(line2), FadeIn(plane_icon))
        self.lecture[2].set_color("#FF00FF")

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN)
        self.wait(1)
