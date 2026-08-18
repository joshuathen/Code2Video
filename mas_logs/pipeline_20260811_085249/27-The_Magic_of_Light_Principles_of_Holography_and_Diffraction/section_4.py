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
        self.setup_layout("Reconstruction: Decoding the Wavefront", [
            "Illuminate the hologram with the reference beam.",
            "The light diffracts through the recorded pattern.",
            "This reconstructs the original 3D wavefront."
        ])
        
        # Use assets as requested
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg]
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg]
        
        hologram = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg", color=BLUE)
        self.place_in_area(hologram, 'C4', 'D6', scale_factor=0.5)
        
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg", color=YELLOW)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFF00")
        self.place_at_grid(laser, 'C2', scale_factor=0.5)
        ref_beam = Line(start=laser.get_right(), end=hologram.get_left(), color="#FFFF00").add_tip()
        self.play(FadeIn(laser), Create(ref_beam))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        diffracted_rays = VGroup(*[
            Line(start=hologram.get_right(), end=hologram.get_right() + np.array([1.5, i * 0.5, 0]), color="#00FFFF")
            for i in [-2, -1, 0, 1, 2]
        ])
        self.play(Create(diffracted_rays))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        reconstructed_obj = Circle(radius=0.5, color="#FF00FF", fill_opacity=0.4)
        self.place_at_grid(reconstructed_obj, 'C6')
        self.play(FadeIn(reconstructed_obj))
        self.wait(2)
