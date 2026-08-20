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
        self.setup_layout("Application: Animal Vocalization Analysis", [
            "Fourier analysis identifies animal vocalization signatures.",
            "Bat chirps translate into unique frequency charts.",
            "Spectrograms reveal patterns used for communication."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show raw bird chirp waveform with bird asset
        t = np.linspace(0, 1, 200)
        y = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)
        waveform = VMobject()
        waveform.set_points_smoothly([np.array([i/200 * 4 - 2, val * 0.5, 0]) for i, val in enumerate(y)])
        
        bird_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bird.svg")
        self.place_at_grid(bird_icon, "A1", scale_factor=0.3)
        
        self.place_in_area(waveform, "A2", "B6", scale_factor=0.7)
        waveform.set_color("#00FF00")
        self.play(Create(waveform), FadeIn(bird_icon))
        self.lecture[0].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Compute FFT to show frequency bars (Simulated)
        freq_bars = VGroup(*[
            Rectangle(height=1 + i*0.5, width=0.3, color="#00FFFF", fill_opacity=0.8)
            for i in range(5)
        ]).arrange(RIGHT, aligned_edge=DOWN)
        self.place_in_area(freq_bars, "C1", "D6", scale_factor=0.6)
        
        self.play(Transform(waveform, freq_bars), FadeOut(bird_icon))
        self.lecture[1].set_color("#00FFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight dominant frequency peak in spectrum associated with the bird
        dominant_peak = freq_bars[4]
        highlight = SurroundingRectangle(dominant_peak, color="#FF0000", buff=0.1)
        
        bird_icon_small = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bird.svg")
        self.place_at_grid(bird_icon_small, "F6", scale_factor=0.2)
        
        self.play(Create(highlight), FadeIn(bird_icon_small))
        self.lecture[2].set_color("#FF0000")
        self.wait(2)
