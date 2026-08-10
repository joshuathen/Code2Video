from manim import *
import numpy as np
import os

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
            "Photography captures only brightness, not wave phase.",
            "Holography interferes object and reference beams.",
            "The film records complex phase interference fringes.",
            "Interference patterns store 3D spatial information.",
            "Object and reference beams must be coherent."
        ]
        self.setup_layout("Holography: Recording the Interference Pattern", lecture_lines)
        
        # Elements
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        mirror = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mirror.svg")
        film = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/film.svg")
        
        plate = Rectangle(width=0.5, height=3, color=WHITE).set_fill(opacity=0.3)
        ref_beam = Line(start=LEFT*2, end=RIGHT*2, color="#FF00FF")
        obj_beam = Line(start=LEFT*2, end=RIGHT*2, color="#FFFF00")
        fringe = Rectangle(width=0.4, height=2.8, color="#00FF00").set_fill(opacity=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.place_at_grid(laser, 'C2', scale_factor=0.5)
        self.play(FadeIn(laser))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        self.place_at_grid(plate, 'C5', scale_factor=1.0)
        self.place_at_grid(ref_beam, 'B5', scale_factor=1.0)
        self.play(FadeIn(plate), Create(ref_beam))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFFF00")
        self.place_at_grid(mirror, 'E1', scale_factor=0.5)
        self.place_at_grid(obj_beam, 'E2', scale_factor=1.0)
        self.play(FadeIn(mirror), Create(obj_beam))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#00FF00")
        self.place_at_grid(fringe, 'D5', scale_factor=1.0)
        self.play(FadeIn(fringe))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFFFF")
        self.place_at_grid(film, 'C6', scale_factor=0.5)
        label1 = Text("Reference", font_size=16).next_to(ref_beam, UP)
        label2 = Text("Object", font_size=16).next_to(obj_beam, DOWN)
        label3 = Text("Interference Pattern", font_size=16).next_to(fringe, RIGHT)
        self.play(FadeIn(film), FadeIn(label1), FadeIn(label2), FadeIn(label3))
        self.wait(2)
