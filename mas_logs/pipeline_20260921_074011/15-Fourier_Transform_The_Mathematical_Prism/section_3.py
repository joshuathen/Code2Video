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
        self.setup_layout("The Core Concept: Time to Frequency", [
            "Time domain shows events occurring.", 
            "Frequency domain reveals signal DNA.", 
            "Integrals match signals to reference frequencies."
        ])
        
        # Assets
        wave = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wave.svg", color=WHITE)
        sine = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sine.svg", color=WHITE)
        freq_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/frequency.svg", color="#FF8800")
        
        comp1 = sine.copy().set_color("#00FFFF")
        comp2 = sine.copy().set_color("#FF00FF")
        sine_components = VGroup(comp1, comp2).arrange(RIGHT, buff=0.2)
        
        # Frequency domain elements
        freq_axis = Line(start=LEFT*2, end=RIGHT*2, color="#FF8800")
        freq_label = Text("Frequency", font_size=16, color="#FF8800").next_to(freq_axis, DOWN)
        freq_plot = VGroup(freq_axis, freq_label, freq_icon)

        # === Animation for Lecture Line 1 ===
        self.place_in_area(wave, 'A4', 'C6', scale_factor=0.6)
        self.play(FadeIn(wave))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.place_in_area(sine_components, 'A4', 'B6', scale_factor=0.5)
        self.play(
            FadeOut(wave),
            FadeIn(sine_components),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        self.wait(1)
        
        # Summing back
        self.play(
            FadeOut(sine_components),
            FadeIn(wave),
            self.lecture[1].animate.set_color(WHITE)
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.place_in_area(freq_plot, 'D3', 'F5', scale_factor=0.75)
        self.lecture[2].set_color("#FF8800")
        self.play(
            FadeIn(freq_plot)
        )
        self.wait(2)
