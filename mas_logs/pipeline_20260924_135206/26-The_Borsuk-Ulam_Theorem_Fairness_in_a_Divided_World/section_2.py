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
        lecture_lines = [
            "Borsuk-Ulam states spheres map to R-n.",
            "Antipodal points share the same value.",
            "Earth has two opposite spots with identical weather."
        ]
        self.setup_layout("The Borsuk-Ulam Theorem", lecture_lines)
        
        # Assets
        earth = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/earth.svg")
        
        # === Animation for Lecture Line 1 ===
        # Borsuk-Ulam states spheres map to R-n.
        self.place_at_grid(earth, "C5", scale_factor=0.6)
        self.play(FadeIn(earth))
        self.play(self.lecture[0].animate.set_color("#3498DB"))

        # === Animation for Lecture Line 2 ===
        # Antipodal points share the same value.
        # Antipodal points
        point_a = Dot(color="#FF0000", radius=0.1)
        point_b = Dot(color="#FF0000", radius=0.1)
        
        # Manually set positions on "surface" of Earth asset
        point_a.move_to(earth.get_center() + np.array([0.5, 0.2, 0]))
        point_b.move_to(earth.get_center() - np.array([0.5, 0.2, 0]))
        
        label_a = Text("x", font_size=20, color=WHITE)
        label_b = Text("-x", font_size=20, color=WHITE)
        self.place_at_grid(label_a, "C6", scale_factor=0.4)
        self.place_at_grid(label_b, "C4", scale_factor=0.4)
        
        self.play(Create(point_a), Create(point_b), Write(label_a), Write(label_b))
        
        # Flash / Pulse to indicate equal value
        self.play(
            point_a.animate.set_color("#00FF00").scale(1.5),
            point_b.animate.set_color("#00FF00").scale(1.5),
            run_time=1
        )
        self.play(
            point_a.animate.set_color("#FF0000").scale(1/1.5),
            point_b.animate.set_color("#FF0000").scale(1/1.5)
        )
        self.play(self.lecture[1].animate.set_color("#E74C3C"))

        # === Animation for Lecture Line 3 ===
        # Earth has two opposite spots with identical weather.
        # Move across surface
        self.play(
            point_a.animate.shift(np.array([-0.3, 0.1, 0])),
            point_b.animate.shift(np.array([0.3, -0.1, 0])),
            label_a.animate.shift(np.array([-0.3, 0.1, 0])),
            label_b.animate.shift(np.array([0.3, -0.1, 0]))
        )
        
        # Flash again
        self.play(
            point_a.animate.set_color("#00FF00"),
            point_b.animate.set_color("#00FF00"),
            run_time=0.5
        )
        
        self.play(self.lecture[2].animate.set_color("#2ECC71"))
        self.wait(2)
