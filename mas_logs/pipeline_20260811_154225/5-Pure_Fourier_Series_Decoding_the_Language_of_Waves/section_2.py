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
        lecture_lines = [
            "Periodic f(x) decomposes into sums of sines and cosines.",
            "Fourier coefficients represent the weight of each frequency.",
            "Think of it like a harmonic light mixer."
        ]
        self.setup_layout("The Core Concept: Defining Fourier Series", lecture_lines)
        
        # Load Assets
        mixer_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mixer.svg")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        axes = Axes(x_range=[-3, 3, 1], y_range=[-1.5, 1.5, 1], x_length=4, y_length=2.5, axis_config={"include_numbers": False})
        graph_square = axes.plot(lambda x: 1 if np.sin(x) >= 0 else -1, color="#FFFFFF", discontinuities=[0, np.pi, -np.pi])
        
        # Asset 1 placement
        mixer_1 = mixer_icon.copy()
        self.place_at_grid(mixer_1, "A5", scale_factor=0.3)
        
        # Fix 29: Place animation in C4-F6
        content_group = VGroup(axes, graph_square)
        self.place_in_area(content_group, 'C4', 'F6', scale_factor=0.8)
        self.play(Create(graph_square), FadeIn(mixer_1))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        sine_wave = axes.plot(lambda x: np.sin(x), color="#FF00FF")
        self.play(Create(sine_wave))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF00")
        sine_wave_3 = axes.plot(lambda x: np.sin(x) + (1/3)*np.sin(3*x), color="#00FF00")
        
        # Asset 2 placement
        mixer_2 = mixer_icon.copy()
        self.place_at_grid(mixer_2, "E5", scale_factor=0.3)
        
        self.play(Transform(sine_wave, sine_wave_3), FadeIn(mixer_2))
        
        self.wait(2)
