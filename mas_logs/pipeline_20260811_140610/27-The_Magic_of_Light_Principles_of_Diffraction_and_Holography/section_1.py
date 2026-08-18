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
            "Light propagates as a traveling wave.",
            "Two overlapping waves create interference.",
            "Constructive interference increases wave intensity.",
            "Destructive interference cancels wave amplitude.",
            "This provides the basis for holography."
        ]
        self.setup_layout("Prerequisite: Wave-Particle Duality and Superposition", lecture_lines)
        
        # Setup colors
        c1, c2, c3, c4, c5 = "#00FFFF", "#FF00FF", "#FFFF00", "#FF4500", "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Show particle wave using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg]
        icon1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hologram.svg", color=c1)
        self.place_at_grid(icon1, 'B5', scale_factor=0.6)
        self.play(FadeIn(icon1))
        self.lecture[0].set_color(c1)

        # === Animation for Lecture Line 2 ===
        wave1 = FunctionGraph(lambda x: 0.5 * np.sin(3 * x), x_range=[-2, 2], color=c2)
        wave2 = FunctionGraph(lambda x: 0.5 * np.sin(3 * x + PI), x_range=[-2, 2], color=c2)
        self.place_at_grid(wave1, 'C4', scale_factor=0.6)
        self.place_at_grid(wave2, 'C5', scale_factor=0.6)
        self.play(Create(wave1), Create(wave2))
        self.lecture[1].set_color(c2)

        # === Animation for Lecture Line 3 ===
        res1 = FunctionGraph(lambda x: 1.0 * np.sin(3 * x), x_range=[-2, 2], color=c3)
        self.place_at_grid(res1, 'D5', scale_factor=0.6)
        self.play(ReplacementTransform(VGroup(wave1, wave2), res1))
        self.lecture[2].set_color(c3)

        # === Animation for Lecture Line 4 ===
        res2 = FunctionGraph(lambda x: 0, x_range=[-2, 2], color=c4)
        self.place_at_grid(res2, 'E4', scale_factor=0.6)
        self.play(Transform(res1, res2))
        self.lecture[3].set_color(c4)

        # === Animation for Lecture Line 5 ===
        # Show the concept of a wave function spreading out using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg]
        icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg", color=c5)
        self.place_at_grid(icon2, 'F5', scale_factor=0.8)
        self.play(FadeIn(icon2), icon2.animate.scale(1.5).set_opacity(0.3))
        self.lecture[4].set_color(c5)
        self.wait(2)
