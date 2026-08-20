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
        self.setup_layout("The Intuition: From Music to Math", [
            "Complex signals are chords of simple sine waves.",
            "Frequency and amplitude build every sound we hear.",
            "Imagine an equalizer breaking down jagged waves."
        ])

        # Colors
        color_simple = "#FF00FF"
        color_complex = "#00FFFF"
        color_peaks = "#FFFF00"

        # --- Setup Objects ---
        
        # Assets
        icon_instrument = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/instrument.svg")
        icon_equalizer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/equalizer.svg")
        
        # 1. Simple wave
        simple_wave = FunctionGraph(lambda x: 0.5 * np.sin(2 * np.pi * x), x_range=[-2, 2], color=color_simple)
        label1 = Text("Simple Tone", font_size=18, color=color_simple)
        self.place_in_area(simple_wave, 'A2', 'B3', scale_factor=0.8)
        self.place_at_grid(label1, 'A1')
        self.place_at_grid(icon_instrument, 'A3', scale_factor=0.3)

        # 2. Complex wave
        complex_wave = FunctionGraph(lambda x: 0.5 * np.sin(2 * np.pi * x) + 0.2 * np.sin(4 * np.pi * x + 1), x_range=[-2, 2], color=color_complex)
        label2 = Text("Complex Signal", font_size=18, color=color_complex)
        self.place_in_area(complex_wave, 'C2', 'D3', scale_factor=0.8)
        self.place_at_grid(label2, 'C1')

        # --- Animations ---

        # === Animation for Lecture Line 1 ===
        self.play(Create(simple_wave), Write(label1), FadeIn(icon_instrument))
        self.lecture[0].set_color(color_simple)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Create(complex_wave), Write(label2))
        self.lecture[1].set_color(color_complex)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Decompose
        wave1 = FunctionGraph(lambda x: 0.5 * np.sin(2 * np.pi * x), x_range=[-1, 1], color=color_simple)
        wave2 = FunctionGraph(lambda x: 0.2 * np.sin(4 * np.pi * x + 1), x_range=[-1, 1], color=color_simple)
        
        self.place_at_grid(icon_equalizer, 'F1', scale_factor=0.3)
        
        self.play(
            FadeOut(complex_wave, label2),
            FadeIn(icon_equalizer),
            TransformFromCopy(simple_wave, wave1),
            ReplacementTransform(simple_wave.copy(), wave2)
        )
        self.place_at_grid(wave1, 'E3', scale_factor=0.7)
        self.place_at_grid(wave2, 'F3', scale_factor=0.7)
        
        self.lecture[2].set_color(color_peaks)
        self.wait(2)
