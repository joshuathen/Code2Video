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
        self.setup_layout("The Core Intuition: The Harmonic Symphony", [
            "Any complex wave is a sum of simple harmonics.",
            "Think of this like stacking musical notes.",
            "Sine waves combine into a square wave.",
            "Frequencies and amplitudes define our unique waveform.",
            "The harmonic symphony reveals hidden structures."
        ])
        
        axes = Axes(x_range=[-PI, PI], y_range=[-1.5, 1.5], axis_config={"include_numbers": False}).scale(0.4)
        self.place_in_area(axes, 'C2', 'F6', scale_factor=0.85)
        
        # Assets
        instrument = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/instrument.svg")
        self.place_at_grid(instrument, 'B3', scale_factor=0.3)
        
        # === Animation for Lecture Line 1 ===
        sine_wave = axes.plot(lambda x: np.sin(x), color="#FFFFFF")
        self.play(Create(sine_wave), FadeIn(instrument), self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        sine2 = axes.plot(lambda x: np.sin(x) + 0.33 * np.sin(3 * x), color="#FFFF00")
        self.play(Transform(sine_wave, sine2), self.lecture[2].animate.set_color("#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Re-highlighting with the instrument icon
        self.play(self.lecture[4].animate.set_color("#00FFFF"), instrument.animate.set_color("#00FFFF"))
        self.wait(1)
