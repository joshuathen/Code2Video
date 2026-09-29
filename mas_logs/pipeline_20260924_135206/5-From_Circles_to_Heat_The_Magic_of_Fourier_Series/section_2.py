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
        self.setup_layout("Fourier Series: Decomposing the Wave", [
            "Fourier series acts as a musical chord.", 
            "Complex signals contain many pure frequencies.", 
            "We sum sine waves of varying amplitudes."
        ])
        
        # Define colors
        color_cyan = "#00FFFF"
        color_yellow = "#FFFF00"
        color_magenta = "#FF00FF"
        color_green = "#00FF00"

        # Asset loading
        guitar_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/guitar.svg")

        # === Animation for Lecture Line 1 ===
        # Display a complex periodic wave; label 'Signal' in #00FFFF.
        signal_axes = Axes(x_range=[0, 6, 1], y_range=[-2, 2, 1], axis_config={"include_tip": False})
        signal = signal_axes.plot(lambda x: np.sin(x) + 0.5 * np.sin(3*x), color=WHITE)
        label = Text("Signal", color=WHITE, font_size=24)
        
        self.place_in_area(signal_axes, "B3", "F6", scale_factor=0.6)
        # Re-plot signal because place_in_area scales the axes/mobjects
        self.place_at_grid(signal, "D4", scale_factor=0.5) 
        self.place_at_grid(label, "C4", scale_factor=0.8)
        self.place_at_grid(guitar_icon, "C6", scale_factor=0.3)
        
        self.play(Create(signal_axes), Create(signal), Write(label), FadeIn(guitar_icon))
        self.lecture[0].set_color(color_cyan)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Overlay three sine waves with varying frequencies.
        wave1 = signal_axes.plot(lambda x: np.sin(x), color=color_yellow)
        wave2 = signal_axes.plot(lambda x: 0.5 * np.sin(3*x), color=color_magenta)
        wave3 = signal_axes.plot(lambda x: 0.25 * np.sin(5*x), color=color_green)
        
        self.play(Create(wave1), Create(wave2), Create(wave3))
        self.lecture[1].set_color(color_yellow)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate the components summing to match the signal, represented by the guitar sound profile.
        sum_wave = VGroup(wave1, wave2, wave3)
        self.play(sum_wave.animate.set_color(WHITE))
        self.lecture[2].set_color(color_green)
        self.wait(2)
