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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Signals are combinations of simple waves.",
            "Time domain shows when signals occur.",
            "Frequency domain shows contained pitches.",
            "Fourier transforms bridge both domains.",
            "Music notes illustrate this duality."
        ]
        self.setup_layout("Prerequisite Intuition: The Fourier Perspective", lecture_lines)
        
        # Elements
        wave = FunctionGraph(lambda x: np.sin(x*3), color=WHITE)
        speaker_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speaker.svg")
        instrument_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/instrument.svg")
        spectral_lines = VGroup(*[Line(UP*0.5, DOWN*0.5, color=YELLOW).shift(RIGHT*i*0.5) for i in range(4)])
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#ADD8E6")
        title_fourier = Text("Fourier Perspective", color="#ADD8E6", font_size=32)
        self.place_at_grid(title_fourier, 'A4', scale_factor=0.9)
        self.play(FadeIn(title_fourier))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        wave_group = VGroup(wave, self.place_at_grid(speaker_icon, 'B1', scale_factor=0.5))
        self.place_in_area(wave_group, 'B4', 'C6', scale_factor=0.6)
        self.play(Create(wave), FadeIn(speaker_icon))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        inst_group = VGroup(spectral_lines, self.place_at_grid(instrument_icon, 'D1', scale_factor=0.5))
        self.place_at_grid(spectral_lines, 'D4', scale_factor=0.7)
        self.play(FadeOut(wave), FadeOut(speaker_icon), FadeIn(spectral_lines), FadeIn(instrument_icon))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FF69B4")
        self.wait(1)
