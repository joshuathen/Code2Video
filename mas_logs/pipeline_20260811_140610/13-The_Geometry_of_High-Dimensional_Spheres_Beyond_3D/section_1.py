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
        lecture_lines = [
            "We begin with 2D circles using the Pythagorean theorem.",
            "Generalizing, we move to 3D spheres.",
            "A sphere is the set of points equidistant from center.",
            "Mathematically, x-squared plus y-squared plus z-squared equals r-squared.",
            "Planar slices reveal how dimensions nest within each other."
        ]
        self.setup_layout("Prerequisite Review: From 2D to 3D", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Create a 2D circle with center at origin, color #FF00FF.
        circle = Circle(radius=1.5, color="#FF00FF")
        self.place_in_area(circle, 'A4', 'D6', scale_factor=0.6)
        self.play(Create(circle))
        self.lecture[0].set_color("#FF00FF")

        # === Animation for Lecture Line 2 ===
        # Transform 2D circle into 3D sphere [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg], color #00FFFF.
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        sphere_asset.set_color("#00FFFF")
        self.place_in_area(sphere_asset, 'A4', 'D6', scale_factor=0.6)
        self.play(Transform(circle, sphere_asset))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Draw a set of equidistant points from center.
        dot = Dot(color=YELLOW)
        self.place_at_grid(dot, 'B3', scale_factor=0.5)
        self.play(Create(dot))
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 4 ===
        # Fade in formula: x^2 + y^2 + z^2 = r^2, color #FFFFFF.
        formula = MathTex("x^2 + y^2 + z^2 = r^2", color=WHITE)
        self.place_at_grid(formula, 'F4', scale_factor=0.7)
        self.play(Write(formula))
        self.lecture[3].set_color(WHITE)

        # === Animation for Lecture Line 5 ===
        # Draw planar slice through center, color #FFFF00.
        slice_rect = Rectangle(width=2, height=0.1, color="#FFFF00", fill_opacity=0.5)
        self.place_in_area(slice_rect, 'A4', 'D6', scale_factor=0.6)
        self.play(Create(slice_rect))
        self.lecture[4].set_color("#FFFF00")
        self.wait(2)
