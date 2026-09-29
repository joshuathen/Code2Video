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

def get_prisoner_mobject(path):
    if os.path.exists(path):
        try:
            return SVGMobject(path, color=WHITE)
        except Exception:
            return Circle(radius=0.3, color=WHITE, fill_opacity=1)
    return Circle(radius=0.3, color=WHITE, fill_opacity=1)

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Hook: The Prisoner's Dilemma", [
            "A warden challenges two prisoners with 64 coins.", 
            "One coin hides a key under the board.", 
            "They must identify the coin to survive."
        ])
        
        svg_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/prisoner.svg"
        alice = get_prisoner_mobject(svg_path)
        bob = get_prisoner_mobject(svg_path)
        
        self.place_at_grid(alice, "B3", scale_factor=0.9)
        self.place_at_grid(bob, "D3", scale_factor=0.9)
        self.play(FadeIn(alice), FadeIn(bob))
        self.lecture[0].set_color("#FFFFFF")

        box = Rectangle(width=2.5, height=1.5, color="#FF00FF")
        self.place_in_area(box, "C4", "C6", scale_factor=0.9)
        self.play(Create(box))
        self.lecture[1].set_color("#FF00FF")

        confess = Text("Confess", font_size=20, color="#00FFFF")
        silent = Text("Silent", font_size=20, color="#00FFFF")
        self.place_at_grid(confess, "C4", scale_factor=0.6)
        self.place_at_grid(silent, "C6", scale_factor=0.6)
        self.play(Write(confess), Write(silent))
        self.lecture[2].set_color("#00FFFF")
        self.wait(2)
