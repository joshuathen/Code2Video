from manim import *
import numpy as np

# Using Cyan as a constant
CYAN_COLOR = "#00FFFF"
MAGENTA_COLOR = "#FF00FF"

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

class Section5Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Application: The Digital World", [
            "Fourier Transform powers modern digital tech.",
            "Frequency identification allows data compression.",
            "We discard imperceptible frequencies efficiently.",
            "Compression reduces file size drastically.",
            "Audio quality remains effectively preserved."
        ])
        
        # Define visual objects
        wave = FunctionGraph(lambda x: np.sin(4*x) + 0.5*np.sin(10*x), x_range=[-3, 3], color=CYAN_COLOR)
        samples = VGroup(*[Dot(point=[x, np.sin(4*x) + 0.5*np.sin(10*x), 0], color=CYAN_COLOR, radius=0.03) for x in np.linspace(-2.5, 2.5, 50)])
        
        # Assets
        mic = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg")
        speaker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speaker.svg")
        
        compressed_samples = VGroup(*[Dot(point=[x, np.sin(4*x), 0], color=MAGENTA_COLOR, radius=0.03) for x in np.linspace(-2.5, 2.5, 20)])
        reconstructed = FunctionGraph(lambda x: np.sin(4*x), x_range=[-3, 3], color=CYAN_COLOR)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(CYAN_COLOR))
        self.place_at_grid(mic, "B2", scale_factor=0.5)
        self.play(FadeIn(mic))
        self.place_in_area(wave, 'C2', 'E4', scale_factor=0.6)
        self.play(Create(wave))
        self.place_in_area(samples, "D3", "F6")
        self.play(Create(samples))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(MAGENTA_COLOR))
        self.play(FadeOut(wave), FadeOut(samples), FadeOut(mic))
        self.place_in_area(compressed_samples, 'C4', 'E6', scale_factor=0.5)
        self.play(Create(compressed_samples))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(CYAN_COLOR))
        self.play(compressed_samples.animate.set_color(CYAN_COLOR))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(MAGENTA_COLOR))
        self.play(compressed_samples.animate.scale(0.7))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(CYAN_COLOR))
        self.place_at_grid(speaker, "B5", scale_factor=0.5)
        self.play(FadeIn(speaker))
        self.place_in_area(reconstructed, 'A3', 'C6', scale_factor=0.7)
        self.play(Create(reconstructed), FadeOut(compressed_samples))
        self.wait(2)
