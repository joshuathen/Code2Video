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
        self.setup_layout("Diffraction: Light Bending Around Obstacles", [
            "Light bends around small obstacles.",
            "This behavior is called diffraction.",
            "Patterns serve as object fingerprints."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Parallel rays
        rays = VGroup(*[Line(UP*1.5 + LEFT*1.5 + RIGHT*i*0.3, UP*1.5 + LEFT*1.5 + RIGHT*i*0.3 + DOWN*1, color=WHITE) for i in range(10)])
        aperture = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/aperture.svg")
        
        self.place_at_grid(rays, 'B2', scale_factor=0.5)
        self.place_at_grid(aperture, 'B4', scale_factor=0.5)
        self.play(FadeIn(rays), FadeIn(aperture))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Wavefronts bending
        arc1 = Arc(radius=0.5, start_angle=PI/2, angle=-PI, color=YELLOW)
        arc2 = Arc(radius=0.8, start_angle=PI/2, angle=-PI, color=YELLOW)
        wavefronts = VGroup(arc1, arc2)
        
        self.place_at_grid(wavefronts, 'D3', scale_factor=0.6)
        self.play(FadeIn(wavefronts))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Fingerprint pattern (placeholder)
        dots = VGroup(*[Dot(radius=0.05, color=BLUE) for _ in range(20)])
        dots.arrange_in_grid(4, 5, buff=0.1)
        
        self.place_at_grid(dots, 'E4', scale_factor=0.7)
        self.play(FadeIn(dots))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
