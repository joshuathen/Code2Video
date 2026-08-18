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
        self.setup_layout("Visual Synthesis & Application", [
            "Building a square wave reveals hidden complexity.",
            "Stacking waves makes the shape sharper.",
            "More frequencies lead to better approximations."
        ])
        
        # Assets
        oscillator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/oscillator.svg")
        speaker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speaker.svg")
        
        # === Animation for Lecture Line 1 ===
        # Show three sine waves stacking together using the [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/oscillator.svg] as the source. Color: #FF00FF.
        wave1 = FunctionGraph(lambda x: 0.5 * np.sin(x), x_range=[-PI, PI])
        wave2 = FunctionGraph(lambda x: 0.3 * np.sin(3*x), x_range=[-PI, PI])
        wave3 = FunctionGraph(lambda x: 0.2 * np.sin(5*x), x_range=[-PI, PI])
        
        waves = VGroup(wave1, wave2, wave3).set_color("#FF00FF")
        oscillator_icon = self.place_at_grid(oscillator, "A2", scale_factor=0.3)
        self.place_in_area(waves, "A4", "C6", scale_factor=0.6)
        self.play(Create(waves), FadeIn(oscillator_icon))
        self.lecture[0].set_color("#FF00FF")

        # === Animation for Lecture Line 2 ===
        # Demonstrate the emergence of a square wave. Color: #00FFFF.
        # Note: Do not use always_redraw for complex/heavy mobjects as per instructions.
        # We will manually transform/create the square wave.
        def square_wave_func(x):
            return 0.5 * np.sin(x) + 0.3 * np.sin(3*x) + 0.2 * np.sin(5*x)

        sum_wave = FunctionGraph(square_wave_func, x_range=[-PI, PI]).set_color("#00FFFF")
        self.place_in_area(sum_wave, "D4", "F6", scale_factor=0.6)
        self.play(Transform(waves, sum_wave))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Highlight the convergence at discontinuities to produce an output signal via [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/speaker.svg]. Color: #FFFF00.
        self.lecture[2].set_color("#FFFF00")
        speaker_icon = self.place_at_grid(speaker, "D2", scale_factor=0.3)
        self.play(FadeIn(speaker_icon))
        
        dot = Dot(color="#FFFF00").move_to(sum_wave.point_from_proportion(0.5))
        self.play(Flash(dot))
        self.wait(1)
