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
        lecture_lines = [
            "Bloch spheres map all possible qubit states.",
            "North pole represents our zero state.",
            "South pole represents our one state.",
            "The surface shows all possible superpositions.",
            "Rotation traces the path between states."
        ]
        self.setup_layout("Visualization: The Bloch Sphere", lecture_lines)
        
        # --- Asset Imports ---
        # Using SVG asset
        sphere_outline = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        sphere_outline.set_color(WHITE)
        
        # --- Animation Setup ---
        # Critic fix: Position sphere in B3-E5
        self.place_in_area(sphere_outline, 'B3', 'E5', scale_factor=0.75)
        
        # Create vector
        vector = Arrow(start=ORIGIN, end=UP * 1.5, color="#00FFFF", buff=0)
        # Critic fix: Position vector, also ensure consistent base
        self.place_in_area(vector, 'C3', 'D4', scale_factor=0.6)
        
        # Critic fix: Labels
        north_label = MathTex(r"|0\rangle", color="#00FF00")
        south_label = MathTex(r"|1\rangle", color="#FF0000")
        
        self.place_at_grid(north_label, 'A4', scale_factor=0.9)
        self.place_at_grid(south_label, 'F4', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(FadeIn(sphere_outline), run_time=1.0)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#00FFFF")
        self.play(FadeIn(vector), Write(north_label), run_time=1.0)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        self.play(Write(south_label), run_time=1.0)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#00FFFF")
        self.play(sphere_outline.animate.set_color("#FFFF00"), run_time=1.0)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#00FFFF")
        # Rotating the vector
        self.play(Rotate(vector, angle=PI, about_point=sphere_outline.get_center()), run_time=2.0)
        self.wait(1)
