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
            "Fourier series can be viewed as rotating gears.",
            "Phasors spin at speeds determined by their frequency.",
            "The path of the final gear traces complex shapes.",
            "Epicycles draw any silhouette with pure harmonic rotations.",
            "Geometry visualizes the math of oscillating signals."
        ]
        self.setup_layout("Visualizing the Series", lecture_lines)
        
        # Load Assets
        gear_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gear.svg")
        
        # Define objects
        wave = SineWave(amplitude=1, frequency=1)
        # Fixes from VideoCritic (Line 59 adjustment / grid usage)
        self.place_in_area(wave, 'D2', 'F5', scale_factor=0.6)
        wave_dot = Dot(color="#FF0000")
        self.place_at_grid(wave_dot, 'E3', scale_factor=0.7)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        gears = VGroup(*[gear_asset.copy().set_color("#FFFFFF") for _ in range(3)])
        gears.arrange(RIGHT, buff=0.2)
        self.place_in_area(gears, 'B2', 'C5', scale_factor=0.5)
        self.play(FadeIn(gears))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FFFF"))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF00FF"))
        self.add(wave, wave_dot)
        self.play(Create(wave), MoveAlongPath(wave_dot, wave), run_time=3, rate_func=linear)
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF0000"))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFFF00"))
        gear_final = gear_asset.copy().set_color("#FFFF00")
        self.place_at_grid(gear_final, 'F4', scale_factor=0.4)
        self.play(FadeIn(gear_final))
        self.wait(1)

class SineWave(VMobject):
    def __init__(self, amplitude=1, frequency=1, **kwargs):
        super().__init__(**kwargs)
        self.set_points_smoothly([
            np.array([x, amplitude * np.sin(frequency * x), 0])
            for x in np.linspace(-2, 2, 100)
        ])
        self.set_color("#FFFF00")
