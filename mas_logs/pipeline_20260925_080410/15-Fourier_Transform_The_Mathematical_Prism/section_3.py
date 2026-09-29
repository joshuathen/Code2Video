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
        lecture_lines = ["Time domain maps signal changes.", "Frequency domain reveals hidden components.", "The integral reveals the spectrum.", "Signals transform into frequency bars.", "Mathematical insight into signal nature."]
        self.setup_layout("The Core Concept: From Time to Frequency", lecture_lines)
        
        # Setup signals
        axes = Axes(x_range=[0, 4, 1], y_range=[-3, 3, 1], axis_config={"include_tip": False}).scale(0.5)
        self.place_in_area(axes, 'A1', 'B6', scale_factor=0.6)
        
        sig1 = axes.plot(lambda t: np.sin(2 * PI * t), color="#FF0000")
        sig2 = axes.plot(lambda t: 0.5 * np.sin(4 * PI * t), color="#00FF00")
        sig3 = axes.plot(lambda t: 0.3 * np.sin(8 * PI * t), color="#00FF00")
        
        complex_signal = axes.plot(lambda t: np.sin(2 * PI * t) + 0.5 * np.sin(4 * PI * t) + 0.3 * np.sin(8 * PI * t), color="#FFFFFF")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF0000"))
        self.play(Create(sig1), Create(sig2), Create(sig3))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        self.play(Transform(VGroup(sig1, sig2, sig3), complex_signal))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FFFF00"))
        # Spectrum representation
        spectrum = VGroup(
            Line(ORIGIN, UP*1, color="#FFFF00").shift(LEFT*0.5),
            Line(ORIGIN, UP*0.5, color="#FFFF00"),
            Line(ORIGIN, UP*0.3, color="#FFFF00").shift(RIGHT*0.5)
        ).arrange(RIGHT, buff=0.2)
        self.place_in_area(spectrum, 'D2', 'F5', scale_factor=0.8)
        self.play(FadeIn(spectrum))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFF00"))
        self.play(Indicate(spectrum))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#808080"))
        self.play(FadeOut(complex_signal))
        self.wait(2)
