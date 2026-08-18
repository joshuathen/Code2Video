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
        self.setup_layout("Prerequisite: The One-Way Street (Hashing)", [
            "Cryptographic hashes create unique digital fingerprints.",
            "The process is strictly a one-way street.",
            "Input data is transformed into a fixed digest.",
            "Reversing the operation is mathematically impossible.",
            "Think of it as a permanent blender effect."
        ])

        # Objects
        title = Text("Hashing: One-way function", font_size=32, color=WHITE)
        self.place_in_area(title, 'A1', 'A3', scale_factor=0.8)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/strawberry.svg]
        strawberry = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/strawberry.svg", color="#FF4500")
        self.place_at_grid(strawberry, 'C2', scale_factor=0.6)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/blender.svg]
        blender = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/blender.svg", color="#8A2BE2")
        blender_label = Text("Hash", font_size=24, color=WHITE)
        blender_group = VGroup(blender, blender_label).arrange(DOWN)
        self.place_at_grid(blender_group, 'C5', scale_factor=0.75)
        
        hash_digest = Text("f7a8...91b2", font_size=48, color="#00FF00")
        self.place_at_grid(hash_digest, 'D6', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(title))
        self.lecture[0].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        self.play(FadeIn(strawberry), Create(blender_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        self.play(strawberry.animate.move_to(self.grid['C5']), run_time=1)
        self.play(FadeOut(strawberry), FadeIn(hash_digest))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFF00")
        self.wait(2)
