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
        lecture_lines = [
            "Map constrained state-space to a geometric graph.",
            "Recursive connections build a Sierpinski Triangle.",
            "Higher disk counts increase fractal depth."
        ]
        self.setup_layout("The Sierpinski Connection", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg]
        # Placing triangle centrally in grid B4-D6 as per constraints to avoid clutter/clipping
        triangle = Polygon(UP, LEFT + DOWN, RIGHT + DOWN, color="#FFD700")
        disk = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg", color="#FFD700")
        self.place_in_area(triangle, "B4", "D6", scale_factor=0.75)
        self.place_at_grid(disk, "B4", scale_factor=0.3)
        self.play(Create(triangle), FadeIn(disk))
        self.lecture[0].set_color("#FFD700")

        # === Animation for Lecture Line 2 ===
        # Iteratively subdivide the triangle into smaller triangles
        v1, v2, v3 = triangle.get_vertices()
        mid12 = (v1 + v2) / 2
        mid23 = (v2 + v3) / 2
        mid31 = (v3 + v1) / 2
        
        t1 = Polygon(v1, mid12, mid31, color="#FFFFFF")
        t2 = Polygon(mid12, v2, mid23, color="#FFFFFF")
        t3 = Polygon(mid31, mid23, v3, color="#FFFFFF")
        
        self.play(
            FadeOut(triangle),
            FadeOut(disk),
            Create(t1), Create(t2), Create(t3)
        )
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        # Show removal of the central triangle
        center_gap = Polygon(mid12, mid23, mid31, color="#000000", fill_opacity=1.0)
        self.play(Create(center_gap))
        self.lecture[2].set_color("#FF4500")
        self.wait(1)
