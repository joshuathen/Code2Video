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
        lecture_lines = [
            "Waves bend around obstacles called diffraction.",
            "Holograms act as complex diffraction gratings.",
            "Diffraction reconstructs original light wavefronts."
        ]
        self.setup_layout("Diffraction: How Light Bends and Encodes", lecture_lines)
        self.lecture.set_opacity(0.5)

        # === Animation for Lecture Line 1 ===
        # Waves bending around a slit [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/slit.svg]
        self.play(self.lecture[0].animate.set_color("#FFFFFF").set_opacity(1.0))
        
        slit = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slit.svg", color=GRAY)
        self.place_at_grid(slit, 'C2', scale_factor=0.7)
        
        waves = VGroup(*[Arc(radius=i*0.3, angle=PI/2, start_angle=-PI/4, color=WHITE) for i in range(1, 6)])
        waves.shift(slit.get_center())
        
        self.add(slit)
        self.play(Create(waves), run_time=2)

        # === Animation for Lecture Line 2 ===
        # Hologram grid
        self.play(self.lecture[1].animate.set_color("#00FFFF").set_opacity(1.0))
        
        hologram_grid = VGroup(*[Line(LEFT, RIGHT, color=BLUE).shift(UP * i * 0.2) for i in range(-5, 6)])
        self.place_in_area(hologram_grid, 'B4', 'D6', scale_factor=0.5)
        
        self.play(FadeIn(hologram_grid))

        # === Animation for Lecture Line 3 ===
        # Reconstructing waves [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg]
        self.play(self.lecture[2].animate.set_color("#FFFF00").set_opacity(1.0))
        
        hologram_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg")
        self.place_at_grid(hologram_asset, 'F4', scale_factor=0.6)
        
        beam = Arrow(LEFT * 1.5, RIGHT * 1.5, color=YELLOW)
        self.place_at_grid(beam, 'E3', scale_factor=0.8)
        
        self.play(beam.animate.shift(RIGHT * 2), FadeIn(hologram_asset))
        self.play(FadeOut(beam))
        self.wait(2)
