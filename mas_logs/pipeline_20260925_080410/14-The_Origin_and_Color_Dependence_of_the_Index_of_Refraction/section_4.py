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
            "Frequency-dependent index causes dispersion.",
            "Different colors refract at angles.",
            "Prisms separate white light components.",
            "Blue bends more than red.",
            "Chromatic dispersion creates visible spectra."
        ]
        self.setup_layout("Application: Chromatic Dispersion", lecture_lines)
        
        # Assets
        glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg")
        prism = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/prism.svg")
        
        # Apply positioning fixes from feedback
        self.place_at_grid(prism, 'B5', scale_factor=0.8)
        
        # Light beams setup
        incident = Line(LEFT*2, ORIGIN, color=WHITE)
        red_ray = Line(ORIGIN, RIGHT*2 + DOWN*0.5, color='#FF0000')
        blue_ray = Line(ORIGIN, RIGHT*2 + UP*0.8, color='#0000FF')
        rays = VGroup(incident, red_ray, blue_ray)
        self.place_in_area(rays, 'C4', 'C6', scale_factor=0.7)
        
        # Label
        spectrum_label = Text("Spectrum", font_size=20, color=WHITE)
        self.place_in_area(spectrum_label, 'A4', 'A6', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color('#FFFF00')
        self.place_at_grid(glass, 'E2', scale_factor=0.5)
        self.play(FadeIn(glass))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color('#00FF00')
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color('#00FFFF')
        self.play(FadeIn(prism), Create(rays))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color('#FF00FF')
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color('#FFA500')
        self.play(FadeIn(spectrum_label))
        self.wait(2)
