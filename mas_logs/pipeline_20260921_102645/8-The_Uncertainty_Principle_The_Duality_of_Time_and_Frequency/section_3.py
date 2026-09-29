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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Precise frequency requires long signal cycles.",
            "Truncating signals causes spectral leakage.",
            "Trade-offs exist in phase information."
        ]
        self.setup_layout("Why Can't We Have Both? The Phase Interference", lecture_lines)
        
        # Assets
        osc1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/oscillator.svg")
        osc2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/oscillator.svg")
        mic = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microphone.svg")
        
        # Animations
        # === Animation for Lecture Line 1 ===
        wave1 = FunctionGraph(lambda x: np.sin(2 * np.pi * 2 * x), x_range=[-2, 2], color=BLUE)
        wave2 = FunctionGraph(lambda x: np.sin(2 * np.pi * 2.2 * x), x_range=[-2, 2], color=GREEN)
        
        pulse_group = VGroup(wave1, wave2, osc1).scale(0.5)
        self.place_in_area(pulse_group, 'A2', 'C5', scale_factor=0.9)
        
        self.play(Create(wave1), Create(wave2), FadeIn(osc1))
        self.lecture[0].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        rect = Rectangle(width=2, height=1, color=RED, fill_opacity=0.2)
        self.place_in_area(rect, 'A2', 'C5', scale_factor=0.95)
        # Using osc2 for the destructive interference demonstration
        self.place_at_grid(osc2, 'D5', scale_factor=0.4)
        
        self.play(Create(rect), FadeIn(osc2))
        self.lecture[1].set_color(RED)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        dot = Dot(color=YELLOW)
        self.place_at_grid(dot, 'B3', scale_factor=0.6)
        self.place_at_grid(mic, 'E3', scale_factor=0.5)
        
        self.play(FadeIn(dot), FadeIn(mic))
        self.lecture[2].set_color(YELLOW)
        self.wait(1)
