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
            "Complex signals are sums of simple sine waves.",
            "See the waveform break into individual clean waves.",
            "Nature's rhythm is built from simple periodic parts."
        ]
        self.setup_layout("The Symphony of Waves", lecture_lines)
        
        # Define the complex wave and its components
        x_range = [0, 4 * PI]
        complex_func = lambda x: np.sin(x) + 0.5 * np.sin(3 * x) + 0.3 * np.sin(5 * x)
        wave_complex = FunctionGraph(complex_func, x_range=x_range, color=WHITE)
        wave1 = FunctionGraph(lambda x: np.sin(x), x_range=x_range, color=BLUE)
        wave2 = FunctionGraph(lambda x: 0.5 * np.sin(3 * x), x_range=x_range, color=GREEN)
        wave3 = FunctionGraph(lambda x: 0.3 * np.sin(5 * x), x_range=x_range, color=RED)
        
        waves = VGroup(wave1, wave2, wave3)
        
        # Load Assets
        icon_tuning = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/tuningfork.svg")
        icon_inst = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/instrument.svg")
        icon_ocean = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ocean.svg")

        # === Animation for Lecture Line 1 ===
        self.place_in_area(wave_complex, 'B3', 'E6', scale_factor=0.5)
        self.place_at_grid(icon_tuning, 'B1', scale_factor=0.4)
        self.play(FadeIn(wave_complex), FadeIn(icon_tuning))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Replace tuning fork with instrument
        self.play(FadeOut(icon_tuning), FadeIn(self.place_at_grid(icon_inst, 'B1', scale_factor=0.4)))
        
        # Explode into parts (refining placement per critical feedback)
        self.place_at_grid(wave_complex, 'C4', scale_factor=0.45)
        self.play(
            ReplacementTransform(wave_complex.copy(), wave1),
            ReplacementTransform(wave_complex.copy(), wave2),
            ReplacementTransform(wave_complex.copy(), wave3),
            wave_complex.animate.set_stroke(opacity=0.3)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Replace instrument with ocean icon
        self.play(FadeOut(icon_inst), FadeIn(self.place_at_grid(icon_ocean, 'B1', scale_factor=0.4)))
        
        # Merge back
        self.place_in_area(wave_complex, 'C3', 'E5', scale_factor=0.4)
        self.play(
            FadeOut(waves),
            wave_complex.animate.set_stroke(opacity=1.0)
        )
        self.wait(2)
