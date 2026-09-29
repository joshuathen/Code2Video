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
            "Signals exist in time and frequency domains.",
            "The Fourier Transform decomposes complex signals.",
            "A guitar pluck illustrates this duality perfectly."
        ]
        self.setup_layout("The Uncertainty Principle: The Fourier Trade-off", lecture_lines)
        
        # Setup Animation elements
        guitar_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/guitar.svg")
        pulse = Dot(color="#FF5733")
        sine = FunctionGraph(lambda x: 0.5 * np.sin(4 * x), x_range=[-2, 2], color="#33FF57")
        overlap_area = Rectangle(width=1, height=1, color="#FFFF33", fill_opacity=0.3)

        # Combine pulse and icon
        pulse_group = VGroup(pulse, guitar_icon.scale(0.3))
        
        # === Animation for Lecture Line 1 ===
        self.place_at_grid(pulse_group, 'A3', scale_factor=0.4)
        self.play(FadeIn(pulse_group))
        self.lecture[0].set_color("#FF5733")
        self.play(pulse_group.animate.shift(RIGHT * 1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_in_area(sine, 'A4', 'B6', scale_factor=0.6)
        self.play(Create(sine))
        self.lecture[1].set_color("#33FF57")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(overlap_area, 'E5', scale_factor=0.5)
        self.lecture[2].set_color("#FFFF33")
        self.play(FadeIn(overlap_area), FadeIn(guitar_icon.copy().move_to(overlap_area)))
        self.wait(2)
