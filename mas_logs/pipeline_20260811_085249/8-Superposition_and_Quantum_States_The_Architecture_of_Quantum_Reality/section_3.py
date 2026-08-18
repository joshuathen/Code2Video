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
            "The Bloch Sphere visualizes quantum states.",
            "The vector points to any surface coordinate.",
            "Rotation around the sphere represents phase."
        ]
        self.setup_layout("Visualization: The Bloch Sphere", lecture_lines)
        
        # Load sphere asset
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        sphere_asset.set_stroke(color=WHITE, width=2)
        
        vector = Arrow(start=ORIGIN, end=UP*1.5, buff=0, color=YELLOW)
        psi_label = MathTex(r"|\psi\rangle", color=YELLOW)
        
        # Grouping
        bloch_group = VGroup(sphere_asset, vector, psi_label)
        
        # Position using revised coordinates
        self.place_in_area(bloch_group, 'B3', 'E5', scale_factor=1.4)
        self.place_at_grid(psi_label, 'A3', scale_factor=1.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(DrawBorderThenFill(sphere_asset))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(BLUE)
        self.play(GrowArrow(vector), Write(psi_label))
        self.play(Rotate(vector, angle=PI/2, axis=RIGHT, about_point=ORIGIN))
        self.play(Rotate(sphere_asset, angle=PI/4, axis=UP))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(BLUE)
        self.play(Rotate(vector, angle=2*PI, axis=UP, about_point=ORIGIN))
        self.wait(1)
