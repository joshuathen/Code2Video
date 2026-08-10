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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Anatomy of a Cone", [
            "A right circular cone forms the base shape.",
            "Introduce the cutting plane slicing through the cone.",
            "Rotate the plane to change the shape created."
        ])
        
        # Define objects
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg").set_color(WHITE)
        plane = Surface(
            lambda u, v: np.array([u, v, 0]),
            u_range=[-1.5, 1.5], v_range=[-1.5, 1.5]
        ).set_color(BLUE).set_opacity(0.5)
        intersection = Circle(radius=1.0, color="#FF00FF")
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg").set_color(YELLOW)
        focus_dot = Dot(color=RED)
        focus_label = Text("Focus", font_size=16, color=RED)
        circle_label = Text("Circle", font_size=16, color="#FF00FF")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.place_in_area(cone, "B4", "E6", scale_factor=0.6)
        self.play(Create(cone))
        
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        self.place_in_area(plane, "B4", "E6", scale_factor=0.5)
        self.play(Create(plane))
        self.place_at_grid(intersection, "B2", scale_factor=0.5)
        self.play(Create(intersection), Write(circle_label.next_to(intersection, RIGHT)))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(YELLOW)
        self.place_at_grid(sphere, "D2", scale_factor=0.4)
        self.play(FadeIn(sphere))
        self.place_at_grid(focus_dot, "B3", scale_factor=0.5)
        self.play(FadeIn(focus_dot), Write(focus_label.next_to(focus_dot, UP)))
        self.wait(2)
