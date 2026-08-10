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
            "Holograms act like complex diffraction gratings.",
            "Reference beams hit fringes to recreate wavefronts.",
            "This reconstructs the original 3D optical field.",
            "The observer perceives depth through diffracted light.",
            "Diffraction gates recreate the object's original shape."
        ]
        self.setup_layout("Reconstruction: The Diffraction Gate", lecture_lines)
        
        # Elements using Assets
        hologram_plate = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plate.svg")
        self.place_at_grid(hologram_plate, 'C4', scale_factor=0.9)
        
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        self.place_at_grid(laser, 'C1', scale_factor=0.5)
        
        beam = Line(start=self.grid["C1"], end=self.grid["C4"], color=RED, stroke_width=4)
        
        diffracted_rays = VGroup(*[\
            Line(start=self.grid["C4"], end=self.grid[pos], color=YELLOW) 
            for pos in ["B5", "C5", "D5"]
        ])
        
        wavefront_label = Text("Reconstructed Wavefront", font_size=20, color=WHITE)
        self.place_at_grid(wavefront_label, 'B3', scale_factor=0.8)
        wavefront_label.set_opacity(0)
        
        grating_label = Text("Complex Diffraction Grating", font_size=20, color=WHITE)
        self.place_at_grid(grating_label, 'E5', scale_factor=0.8)
        grating_label.set_opacity(0)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(hologram_plate))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(laser), Create(beam))
        self.lecture[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        self.play(Create(diffracted_rays))
        self.lecture[2].set_color("#FFFF00")

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        grating_label.set_opacity(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        wavefront_label.set_opacity(1)
        
        self.wait(2)
