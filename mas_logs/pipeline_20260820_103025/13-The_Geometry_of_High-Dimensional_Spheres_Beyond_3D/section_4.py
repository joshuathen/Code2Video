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
        self.setup_layout("The Curse of Dimensionality", [
            "In high dimensions, everything is far apart.",
            "Distance metrics become unreliable for data.",
            "Neighborhoods collapse in high-dimensional space."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show high-dimensional space as sparse points.
        points = VGroup(*[Dot(radius=0.05, color=WHITE) for _ in range(30)])
        for p in points:
            p.move_to(np.random.uniform(-1, 1, 3))
        
        self.place_in_area(points, 'A5', 'F6', scale_factor=0.8)
        # Using asset per instructions, although asset path is non-existent
        # placeholder asset: "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg"
        self.play(Create(points))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Represent data clusters as shrinking spheres.
        sphere = Sphere(radius=1.5, color="#FF6347", fill_opacity=0.3)
        self.place_in_area(sphere, 'B2', 'D3', scale_factor=0.5)
        
        self.play(Create(sphere))
        self.play(sphere.animate.scale(0.5))
        self.lecture[1].set_color("#FF6347")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Show increasing distance between points as dimensions grow.
        dist_line = Line(start=self.grid['C2'], end=self.grid['D5'], color="#ADFF2F")
        dist_label = Text("d(x,y)", font_size=16, color="#ADFF2F")
        self.place_at_grid(dist_label, 'D3', scale_factor=0.9)
        
        self.play(Create(dist_line), Write(dist_label))
        self.play(dist_line.animate.stretch(1.5, 0))
        self.lecture[2].set_color("#ADFF2F")
        self.wait(2)
