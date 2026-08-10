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
            "Coherent light maintains a stable phase relationship.",
            "Superposition creates fixed interference nodes and anti-nodes.",
            "Path difference between waves determines constructive interference."
        ]
        self.setup_layout("Prerequisite: The Wave Nature of Light", lecture_lines)

        # Assets
        source_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/source.svg"
        laser_svg = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg"
        
        source1 = SVGMobject(source_svg)
        source2 = SVGMobject(source_svg)
        self.place_at_grid(source1, "C2", scale_factor=0.3)
        self.place_at_grid(source2, "D2", scale_factor=0.3)
        
        # Waves
        waves1 = SVGMobject(laser_svg)
        waves2 = SVGMobject(laser_svg)
        wave_group = VGroup(waves1, waves2)
        self.place_at_grid(wave_group, "D4", scale_factor=0.6)

        # Labels
        constructive_label = Text("Constructive", font_size=20, color=YELLOW)
        destructive_label = Text("Destructive", font_size=20, color=GREEN)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFFF"), FadeIn(source1), FadeIn(source2))

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[1].animate.set_color("#FF00FF"),
            FadeIn(wave_group)
        )

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[2].animate.set_color("#FFFF00"),
            FadeIn(self.place_at_grid(constructive_label, 'D5', scale_factor=0.5)),
            FadeIn(self.place_at_grid(destructive_label, 'E5', scale_factor=0.5))
        )
