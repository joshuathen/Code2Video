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
        self.setup_layout("The Phenomenon of Diffraction", [
            "Diffraction is light bending around obstacles.",
            "Light waves overlap to form interference patterns.",
            "Interference patterns map intensity peaks and troughs."
        ])
        
        # Animation 1: Obstacle with small gap
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        obstacle = VGroup(
            Line(UP * 2, UP * 0.2),
            Line(DOWN * 2, DOWN * 0.2)
        ).set_color("#8A2BE2")
        self.place_in_area(obstacle, 'B4', 'E6', scale_factor=0.6)
        self.play(Create(obstacle))
        self.lecture[0].set_color("#8A2BE2")
        self.wait(1)

        # Animation 2: Propagate plane waves towards gap
        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1)
        waves = VGroup(*[
            Line(LEFT * 1.0, RIGHT * 1.0).set_color("#00FF7F").shift(UP * i * 0.2)
            for i in range(-5, 6)
        ])
        self.place_at_grid(waves, 'C4', scale_factor=0.7)
        self.play(FadeIn(waves))
        self.play(waves.animate.shift(RIGHT * 1.5), run_time=2)
        self.lecture[1].set_color("#00FF7F")
        self.wait(1)

        # Animation 3: Illustrate diffraction pattern after gap
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1)
        pattern = VGroup(*[
            Arc(radius=i*0.25, start_angle=-PI/4, angle=PI/2).set_color("#FFD700")
            for i in range(1, 6)
        ])
        self.place_at_grid(pattern, 'D5', scale_factor=0.6)
        self.play(Create(pattern))
        self.lecture[2].set_color("#FFD700")
        self.wait(2)
