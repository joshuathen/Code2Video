from manim import *
import numpy as np
import os
from pathlib import Path

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
        # Setup Scene
        lecture_lines = [
            "In a vacuum, light travels at its maximum speed.",
            "Inside materials, molecules slow down the light's progress.",
            "The refractive index 'n' calculates this speed reduction."
        ]
        self.setup_layout("Prerequisite Knowledge: Optical Density and Speed", lecture_lines)

        # Colors
        BLUE_GLASS = "#4682B4"
        GREY_MOL = "#808080"
        PHOTON_YELLOW = YELLOW

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(PHOTON_YELLOW))
        
        # Photon Character
        photon = Dot(color=PHOTON_YELLOW, radius=0.15)
        self.place_at_grid(photon, "C1")
        
        photon_label = Text("Photon", font_size=16, color=PHOTON_YELLOW)
        photon_label.add_updater(lambda m: m.next_to(photon, UP, buff=0.1))

        # Movement through Vacuum (Fast)
        self.add(photon, photon_label)
        self.play(
            photon.animate.move_to(self.grid["C3"]),
            run_time=1.0,
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(BLUE_GLASS)
        )

        # Glass asset integration [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/glass.svg]
        glass_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/glass.svg"
        if os.path.exists(glass_asset_path):
            glass_mobject = SVGMobject(glass_asset_path)
            glass_mobject.set_color(BLUE_GLASS)
            glass_mobject.set_fill(BLUE_GLASS, opacity=0.3)
        else:
            # Fallback
            glass_mobject = Rectangle(width=3.2, height=5.5, fill_color=BLUE_GLASS, fill_opacity=0.3, stroke_width=0)
        
        self.place_in_area(glass_mobject, "A4", "F6")
        
        # Medium label repositioned per Issue 33
        medium_label = Text("Glass Medium", font_size=18, color=BLUE_GLASS)
        self.place_in_area(medium_label, 'E4', 'F6', scale_factor=0.8)

        # Molecules (the 'crowd')
        molecule_spots = ["A4", "A6", "B5", "C4", "C6", "D5", "E4", "E6", "F5"]
        molecules = VGroup(*[Circle(radius=0.1, color=GREY_MOL, fill_opacity=0.8) for _ in molecule_spots])
        for m, pos in zip(molecules, molecule_spots):
            self.place_at_grid(m, pos)

        self.play(FadeIn(glass_mobject), FadeIn(medium_label), FadeIn(molecules))

        # Movement through Glass (Slow)
        target_point = self.grid["C6"]
        self.play(
            photon.animate.move_to(target_point),
            run_time=3.5,
            rate_func=linear
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(WHITE)
        )

        # Formula Display with scale fix per Issue 32
        formula = Text("n = c / v", color=WHITE)
        self.place_in_area(formula, 'A1', 'B3', scale_factor=1.0)

        # Formula Labels with scale and area fix per Issue 34
        n_desc = Text("n: Refractive Index", font_size=20, color=WHITE)
        c_desc = Text("c: Speed in Vacuum", font_size=20, color=WHITE)
        v_desc = Text("v: Speed in Medium", font_size=20, color=WHITE)
        
        labels_vgroup = VGroup(n_desc, c_desc, v_desc).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        self.place_in_area(labels_vgroup, 'D1', 'F3', scale_factor=0.9)

        self.play(Write(formula))
        self.play(FadeIn(labels_vgroup))
        self.wait(3)
