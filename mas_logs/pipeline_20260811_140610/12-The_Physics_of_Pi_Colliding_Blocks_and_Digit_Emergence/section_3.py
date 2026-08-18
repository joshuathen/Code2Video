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
        self.setup_layout("Geometric Mapping: From Collisions to Arcs", [
            "Velocity states trace an elliptical arc.",
            "Specific mass ratios turn ellipses circular.",
            "Collision counts correlate to circle arc length.",
            "Pi emerges from the geometry of motion.",
            "Arc length reflects the number of bounces."
        ])

        # Assets
        vector_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg")
        arc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/arc.svg")

        # === Animation for Lecture Line 1 ===
        ellipse = Ellipse(width=3, height=1.5, color="#FF00FF")
        self.place_in_area(ellipse, "A1", "C6", scale_factor=0.8)
        self.place_at_grid(vector_icon, "B3", scale_factor=0.5)
        self.play(Create(ellipse), FadeIn(vector_icon))
        self.lecture[0].set_color("#FF00FF")

        # === Animation for Lecture Line 2 ===
        circle = Circle(radius=1.2, color="#00FFFF")
        self.place_in_area(circle, "A1", "C6", scale_factor=0.8)
        self.play(Transform(ellipse, circle))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        arc = Arc(radius=1.2, angle=PI/2, color="#98FB98")
        self.place_in_area(arc, "A1", "C6", scale_factor=0.8)
        self.play(Create(arc))
        self.lecture[2].set_color("#98FB98")

        # === Animation for Lecture Line 4 ===
        pi_label = MathTex(r"\\pi", color="#FFD700")
        self.place_at_grid(pi_label, "D3", scale_factor=2)
        self.play(Write(pi_label))
        self.lecture[3].set_color("#FFD700")

        # === Animation for Lecture Line 5 ===
        self.place_at_grid(arc_icon, "D5", scale_factor=0.5)
        self.play(FadeIn(arc_icon), arc.animate.set_color("#FF6347"))
        self.lecture[4].set_color("#FF6347")
        self.wait(2)
