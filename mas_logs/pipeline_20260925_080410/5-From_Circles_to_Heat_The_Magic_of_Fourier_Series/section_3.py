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
            "Heat spreads in a metal rod.",
            "Fourier series solves heat diffusion.",
            "Each harmonic shows temperature decay.",
            "Sharp spots smooth out quickly.",
            "Heat follows predictable wave math."
        ]
        self.setup_layout("The Heat Equation Bridge", lecture_lines)

        # Load rod asset
        rod = SVGMobject('/scratch/pawsey1357/jthen/Code2Video/assets/icon/rod.svg')
        self.place_in_area(rod, 'B2', 'B5', scale_factor=0.7)
        
        # Color initialization
        rod.set_color('#FF8000')

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color('#FF8000')
        self.play(FadeIn(rod))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color('#FFFFFF')
        fourier_curves = VGroup(*[
            FunctionGraph(lambda x: 0.1 * np.sin(k * np.pi * x), x_range=[-1, 1], color=WHITE)
            for k in range(1, 4)
        ]).arrange(DOWN, buff=0.1)
        self.place_in_area(fourier_curves, 'D2', 'E5', scale_factor=0.6)
        self.play(FadeIn(fourier_curves))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color('#FFFF00')
        self.play(fourier_curves.animate.set_color('#FFFF00'))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color('#00FFFF')
        self.play(rod.animate.set_color_by_gradient('#FF0000', '#00FFFF'))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color('#00FF00')
        final_state = Line(start=LEFT, end=RIGHT, color='#00FF00')
        self.place_in_area(final_state, 'B2', 'B5', scale_factor=0.5)
        self.play(ReplacementTransform(rod, final_state))
        self.wait(1)
