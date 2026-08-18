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
        self.setup_layout("Prerequisite: The Nature of Wave Interference", [
            "Light waves carry both amplitude and phase.",
            "Coherent lasers maintain constant phase relationships.",
            "Wave superposition creates unique interference patterns."
        ])
        self.lecture.set_opacity(0)

        # Assets
        laser1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        laser2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        
        wave1 = FunctionGraph(lambda x: 0.5 * np.sin(2 * np.pi * x), x_range=[-2, 2], color="#FFFF00")
        wave2 = FunctionGraph(lambda x: 0.5 * np.sin(2 * np.pi * x), x_range=[-2, 2], color="#FFFF00")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        # Place waves (fixed vertical stacking/congestion)
        self.place_in_area(wave1, 'A2', 'B5', scale_factor=0.6)
        self.place_in_area(wave2, 'C2', 'D5', scale_factor=0.6)
        # Place Asset
        self.place_at_grid(laser1, 'A1', scale_factor=0.3)
        self.play(Create(wave1), Create(wave2), FadeIn(laser1))
        self.lecture[0].set_color("#FFFF00")
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_opacity(1)
        constructive = FunctionGraph(lambda x: np.sin(2 * np.pi * x), x_range=[-2, 2], color="#00FF00")
        # Place constructive pattern (fixed overlap)
        self.place_in_area(constructive, 'A2', 'B5', scale_factor=0.8)
        self.play(
            Transform(wave1, constructive),
            FadeOut(wave2),
            self.lecture[1].animate.set_color("#00FF00")
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_opacity(1)
        destructive = Line(start=np.array([-2, 0, 0]), end=np.array([2, 0, 0]), color="#FF0000")
        # Place destructive pattern (fixed overlap and underutilization)
        self.place_in_area(destructive, 'E2', 'F5', scale_factor=0.8)
        # Place Asset
        self.place_at_grid(laser2, 'F1', scale_factor=0.3)
        self.play(
            FadeIn(laser2),
            Transform(wave1, destructive),
            self.lecture[2].animate.set_color("#FF0000")
        )
        self.wait(2)
