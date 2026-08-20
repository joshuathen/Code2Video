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
        self.setup_layout("Prerequisites: Signals and Sequences", [
            "A discrete signal is just a sequence of numbers.", 
            "'Discrete' means individual steps, not a continuous flow.", 
            "Think of sound as a list of volume levels."
        ])
        
        # Load Assets
        mic = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg")
        speaker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speaker.svg")
        
        # Setup Animation Elements
        # Signal pulse
        pulse = FunctionGraph(lambda x: np.sin(x*3), x_range=[-1.5, 1.5], color="#00FF00")
        self.place_in_area(pulse, 'A4', 'C6', scale_factor=0.6)
        
        # Dots
        dots = VGroup(*[Dot(color="#FF6600") for _ in range(5)])
        self.place_in_area(dots, 'D4', 'F6', scale_factor=0.7)
        
        # Sampling Composite
        sampling_wave = FunctionGraph(lambda x: np.sin(x*3), x_range=[-1.5, 1.5], color="#FFFFFF")
        sampling_dots = VGroup(*[Dot(color="#FFFFFF", radius=0.08) for _ in range(5)])
        sampling = VGroup(sampling_wave, sampling_dots)
        self.place_in_area(sampling, 'A4', 'C6', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        self.place_at_grid(mic, 'B2', scale_factor=0.5)
        self.play(FadeIn(mic), Create(pulse))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF6600")
        # Ensure dots are positioned relative to grid
        for i, dot in enumerate(dots):
            dot.move_to(self.grid[f"D{i+2}"])
        self.play(FadeIn(dots))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFFFF")
        self.place_at_grid(speaker, 'E2', scale_factor=0.5)
        self.play(
            ReplacementTransform(pulse, sampling_wave), 
            ReplacementTransform(dots, sampling_dots),
            FadeIn(speaker)
        )
        self.wait(2)
