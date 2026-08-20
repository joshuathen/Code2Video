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
        self.setup_layout("Linear Combinations and Span", [
            "Linear combination: c1v1 + c2v2.",
            "Span is the set of all reachable points.",
            "Vectors A and B fill the 2D plane."
        ])

        # Define Axes
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True}).scale(0.5)
        self.place_at_grid(axes, 'D2', scale_factor=0.6)
        self.add(axes)

        # Assets
        plane_bg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg")
        self.place_in_area(plane_bg, 'B3', 'E5', scale_factor=0.5)
        plane_bg.set_fill(color="#00FFFF", opacity=0.3)

        # Vectors
        v = Vector([1, 0], color="#FF5733")
        w = Vector([0, 1], color="#33FF57")
        # Ensure vectors are properly placed relative to axes
        v.shift(axes.c2p(0, 0))
        w.shift(axes.c2p(0, 0))
        self.add(v, w)

        # Point marker for reachability
        points = Dot(color=YELLOW)
        self.place_at_grid(points, 'D3', scale_factor=0.4)
        points.move_to(axes.c2p(0, 0))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF5733")
        self.play(Create(v), Create(w))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.play(FadeIn(plane_bg))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#33FF57")
        self.play(points.animate.move_to(axes.c2p(1, 1)))
        self.play(points.animate.move_to(axes.c2p(-1, 0.5)))
        self.play(points.animate.move_to(axes.c2p(0, -1)))
        self.wait(1)
