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
        self.setup_layout("Prerequisite: The Bit vs. The Qubit", [
            "Classical bits are binary: 0 or 1.",
            "Quantum bits, or qubits, hold superposition.",
            "A qubit is a point on a sphere."
        ])
        
        # Assets
        asset_sphere = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg"
        
        # Elements
        bit = SVGMobject(asset_sphere).scale(0.5)
        bloch_sphere = SVGMobject(asset_sphere).scale(1.2)
        zero_label = Tex(r"$|0\rangle$").scale(0.9)
        one_label = Tex(r"$|1\rangle$").scale(0.9)

        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg]
        self.place_at_grid(bit, 'C3')
        bit.set_color("#FFFFFF")
        self.play(FadeIn(bit))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg]
        self.lecture[1].set_color("#00FFFF")
        self.play(FadeOut(bit))
        # Fix: Line 61: self.place_at_grid(sphere, 'D4', scale_factor=1.2)
        self.place_at_grid(bloch_sphere, 'D4', scale_factor=1.2)
        bloch_sphere.set_color("#00FFFF")
        self.play(FadeIn(bloch_sphere))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg]
        self.lecture[2].set_color("#00FF00")
        # Fix: Line 62: self.place_at_grid(label_0, 'D3', scale_factor=0.9); self.place_at_grid(label_1, 'D5', scale_factor=0.9)
        self.place_at_grid(zero_label, 'D3', scale_factor=0.9)
        self.place_at_grid(one_label, 'D5', scale_factor=0.9)
        
        self.play(Write(zero_label), Write(one_label))
        
        # Superposition vector
        vector = Line(start=ORIGIN, end=UP*1, color="#00FF00").move_to(bloch_sphere.get_center())
        self.play(Create(vector))
        self.play(Rotate(vector, angle=PI/2, about_point=bloch_sphere.get_center()))
        self.wait(2)
