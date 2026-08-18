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
        self.setup_layout("Reconstruction: Decoding the 3D Image", [
            "Reference beam illuminates the recorded hologram.",
            "Diffraction pattern bends the light beam.",
            "Original wavefronts emerge from the plate.",
            "Parallax creates a true 3D image.",
            "We perceive the object in space."
        ])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1.0)
        self.lecture[0].set_color("#FFFF00")
        
        # Assets: Plate
        plate = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plate.svg", color=WHITE)
        self.place_in_area(plate, 'B2', 'D2', scale_factor=1.0)
        
        beam = Line(start=LEFT*2, end=RIGHT*2, color="#FFFF00").next_to(plate, LEFT)
        
        self.play(FadeIn(plate), Create(beam))
        self.play(beam.animate.shift(RIGHT*4), run_time=2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1.0)
        self.lecture[1].set_color("#00FF00")
        diffracted_waves = VGroup(*[Arc(radius=0.3 + i*0.2, angle=PI/2, color="#00FF00") for i in range(3)])
        diffracted_waves.arrange(RIGHT, buff=0.1).next_to(plate, RIGHT)
        self.play(Create(diffracted_waves))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1.0)
        self.lecture[2].set_color("#FFFFFF")
        
        # Assets: Hologram (as object)
        cat = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg", color=WHITE)
        self.place_at_grid(cat, 'C6', scale_factor=0.7)
        self.play(FadeIn(cat))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_opacity(1.0)
        self.lecture[3].set_color("#00FFFF")
        
        # Assets: Eye
        viewer_eye = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/eye.svg", color="#00FFFF")
        self.place_at_grid(viewer_eye, 'E4', scale_factor=0.6)
        
        self.play(FadeIn(viewer_eye))
        self.play(viewer_eye.animate.shift(RIGHT*0.5), viewer_eye.animate.shift(LEFT*0.5), run_time=2)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_opacity(1.0)
        self.lecture[4].set_color("#FF00FF")
        
        # Assets: Glow (using hologram asset again as per instructions)
        glow = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg", color="#FF00FF")
        self.place_at_grid(glow, 'C6', scale_factor=0.8)
        glow.set_opacity(0.5)
        
        self.play(FadeIn(glow))
        self.wait(2)
