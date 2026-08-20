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
        self.setup_layout("Visual Proof: The Ellipse Case", [
            "Points on an ellipse follow a constant sum distance.",
            "Dandelin spheres bridge this geometry to algebra.",
            "The sphere tangency points are the ellipse foci.",
            "Distance to foci is constant for every point.",
            "This explains the definition of an ellipse."
        ])
        
        # Load Assets
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg")
        sphere1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        sphere2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        
        # Setup Geometry group
        geometry_group = VGroup(cone, sphere1, sphere2)
        
        foci = VGroup(Dot(color="#FFD700"), Dot(color="#FFD700"))
        foci_label = Text("Foci", font_size=20, color="#FFD700")

        # === Animation for Lecture Line 1 ===
        self.place_in_area(cone, 'C3', 'F6', scale_factor=0.7)
        self.play(FadeIn(cone))
        self.lecture[0].set_color("#00FF00")
        
        # === Animation for Lecture Line 2 ===
        self.place_at_grid(sphere1, 'B3', scale_factor=0.5)
        self.place_at_grid(sphere2, 'D3', scale_factor=0.5)
        self.play(FadeIn(sphere1), FadeIn(sphere2))
        self.lecture[1].set_color(BLUE)
        
        # === Animation for Lecture Line 3 ===
        self.place_at_grid(foci[0], 'B4', scale_factor=0.8)
        self.place_at_grid(foci[1], 'D4', scale_factor=0.8)
        self.place_at_grid(foci_label, 'B5', scale_factor=0.6)
        self.play(FadeIn(foci), Write(foci_label))
        self.lecture[2].set_color("#FFD700")
        
        # === Animation for Lecture Line 4 ===
        # Using the geometry group defined earlier
        self.place_in_area(geometry_group, 'C3', 'F6', scale_factor=0.7)
        self.play(FadeIn(geometry_group))
        self.lecture[3].set_color(WHITE)
        
        # === Animation for Lecture Line 5 ===
        self.play(Flash(geometry_group, color=WHITE))
        self.lecture[4].set_color(WHITE)
        self.wait(2)
