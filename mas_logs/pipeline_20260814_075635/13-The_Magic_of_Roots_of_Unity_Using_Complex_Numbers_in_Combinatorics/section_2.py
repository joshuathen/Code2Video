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
        self.setup_layout("The Roots of Unity Filter Lemma", [
            "Consider the n-th roots of unity.",
            "Sum their powers: 1/n times the sum.",
            "Cancel all terms except where n divides k."
        ])

        # === Animation for Lecture Line 1 ===
        # Visualize roots of unity on complex plane using SVGMobject asset. (#00FF00)
        # Note: Asset path used directly as requested.
        circle = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg", color=WHITE)
        axes = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False})
        roots = [np.exp(2j * PI * k / 4) for k in range(4)]
        dots = VGroup(*[Dot(axes.c2p(r.real, r.imag), color="#00FF00") for r in roots])
        
        root_group = VGroup(circle, axes, dots)
        self.place_in_area(root_group, "B2", "E5", scale_factor=0.55)
        
        self.play(FadeIn(circle), Create(axes))
        self.play(FadeIn(dots))
        self.lecture[0].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Animate summation over roots of unity. (#FFFF00)
        vectors = VGroup(*[Arrow(start=ORIGIN, end=dots[i].get_center(), color="#FFFF00", buff=0) for i in range(4)])
        self.play(Create(vectors))
        self.lecture[1].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show destructive interference canceling unwanted terms. (#FF00FF)
        self.play(FadeOut(vectors))
        # Visualizing cancellation
        cancellation = Text("Sum = 0", color="#FF00FF").scale(0.8)
        self.place_at_grid(cancellation, "D5", scale_factor=0.7)
        
        self.play(Write(cancellation))
        self.lecture[2].set_color("#FF00FF")
        self.wait(2)
