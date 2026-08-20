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
        self.setup_layout("Scalar Multiplication: Stretching Space", [
            "Scalars stretch or shrink vectors.",
            "They change length, maintaining orientation.",
            "Negative scalars reverse the direction."
        ])
        
        # Setup vectors and assets
        vec_base = np.array([1, 1, 0])
        
        spring = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/spring.svg")
        mirror = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        vec1 = Arrow(start=ORIGIN, end=vec_base, buff=0, color="#00FFFF")
        self.place_at_grid(vec1, 'C5', scale_factor=0.7)
        self.place_at_grid(spring, 'B5', scale_factor=0.5)
        self.play(Create(vec1), FadeIn(spring))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFD700")
        vec2 = Arrow(start=ORIGIN, end=vec_base * 2, buff=0, color="#FFD700")
        self.place_at_grid(vec2, 'D5', scale_factor=0.7)
        self.play(ReplacementTransform(vec1, vec2), FadeOut(spring))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        vec3 = Arrow(start=ORIGIN, end=vec_base * -1, buff=0, color="#FF4500")
        self.place_at_grid(vec3, 'E5', scale_factor=0.7)
        self.place_at_grid(mirror, 'F5', scale_factor=0.5)
        self.play(ReplacementTransform(vec2, vec3), FadeIn(mirror))
        self.wait(1)
