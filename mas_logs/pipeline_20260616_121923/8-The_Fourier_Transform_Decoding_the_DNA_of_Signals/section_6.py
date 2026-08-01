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

class Section6Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        lecture_lines = [
            'The Fourier Transform reveals the hidden DNA of signals.',
            'It splits complex waves into their pure components.',
            'A powerful lens for understanding our physical universe.'
        ]
        self.setup_layout("Summary and Conclusion", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color(YELLOW)

        # Blue Prism (Triangle)
        prism = Polygon(
            [-1, -0.866, 0], [1, -0.866, 0], [0, 0.866, 0],
            color="#00AAFF", stroke_width=4, fill_opacity=0.4
        ).set_fill("#00AAFF")
        # Fix Issue 36: Scale adjusted to 1.0
        self.place_in_area(prism, 'C3', 'D4', scale_factor=1.0)

        # Complex White Wave
        # Composed of 3 frequencies
        def wave_func(x):
            return 0.4 * (np.sin(3 * x) + 0.5 * np.sin(7 * x) + 0.3 * np.sin(15 * x))
        
        input_wave = FunctionGraph(
            wave_func,
            x_range=[-1.5, 1.5],
            color="#FFFFFF",
            stroke_width=3
        )
        # Fix Issue 35: Positioned at C2-D2 and scaled to 0.7
        self.place_in_area(input_wave, 'C2', 'D2', scale_factor=0.7)

        self.play(Create(prism), Create(input_wave))
        self.wait(0.5)
        
        # Wave "travels" into the prism
        self.play(
            input_wave.animate.move_to(prism.get_center()).set_opacity(0),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Update highlighting
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Distinct frequency lines (Rainbow)
        rainbow_colors = [RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE]
        spectrum_lines = VGroup()
        for i, color in enumerate(rainbow_colors):
            # Heights representing different amplitudes
            h = 1.8 - (i * 0.2)
            line = Line(ORIGIN, UP * h, color=color, stroke_width=8)
            spectrum_lines.add(line)
        
        spectrum_lines.arrange(RIGHT, buff=0.25)
        self.place_in_area(spectrum_lines, 'C5', 'D6', scale_factor=0.9)

        # Spectrum lines emerge from prism
        self.play(
            LaggedStart(
                *[FadeIn(line, shift=line.get_center() - prism.get_center()) for line in spectrum_lines],
                lag_ratio=0.15
            ),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Update highlighting
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # DNA Text
        dna_text = Text("Decoding the DNA of Signals", color="#FFFFFF", font_size=28)
        # Fix Issue 37: Area narrowed to F2-F5, scale 0.8
        self.place_in_area(dna_text, 'F2', 'F5', scale_factor=0.8)

        self.play(Write(dna_text))
        # Text expands
        self.play(dna_text.animate.scale(1.25), run_time=1.5)
        self.wait(4)
