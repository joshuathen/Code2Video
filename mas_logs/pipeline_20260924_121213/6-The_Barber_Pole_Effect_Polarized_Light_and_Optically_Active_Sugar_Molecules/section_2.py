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
        lecture_lines = ["Sugar molecules are chiral and asymmetric.", "They force light into spiral paths.", "This action rotates the light's plane."]
        self.setup_layout("The Concept of Optical Activity", lecture_lines)
        
        # Assets
        molecule = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/molecule.svg")
        light_wave = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg")
        sample = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sample.svg")
        rotation_label = MathTex(r"\\alpha", color="#FFD700", font_size=36)
        
        # === Animation for Lecture Line 1 ===
        # Sugar molecules are chiral and asymmetric
        self.place_at_grid(molecule, 'B3', scale_factor=0.7)
        self.place_at_grid(sample, 'C4', scale_factor=0.7)
        self.play(FadeIn(molecule), FadeIn(sample))
        self.lecture[0].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # They force light into spiral paths
        self.place_at_grid(light_wave, 'C5', scale_factor=0.7)
        self.play(Create(light_wave))
        self.play(light_wave.animate.rotate(PI/4, about_point=self.grid['C5']))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This action rotates the light's plane
        self.place_at_grid(rotation_label, 'D5', scale_factor=1.0)
        self.play(FadeIn(rotation_label))
        self.lecture[2].set_color("#00FF00")
        self.wait(2)
