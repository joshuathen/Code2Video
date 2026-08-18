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
        lecture_lines = [
            "Observation causes a 'wavefunction collapse'.",
            "The probability wave sharpens into one point.",
            "Interaction with measurement forces a classical reality.",
            "Schrödinger’s Cat is a cloud of states.",
            "Opening the box forces a choice."
        ]
        self.setup_layout("The Measurement Problem", lecture_lines)
        
        # Mobjects
        wave = FunctionGraph(lambda x: 0.5 * np.sin(3 * x), x_range=[-2, 2], color=WHITE)
        dot = Dot(color=WHITE)
        bell = FunctionGraph(lambda x: 1 * np.exp(-x**2), x_range=[-2, 2], color=YELLOW)
        probe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/probe.svg", color=RED)
        box = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/box.svg", color=WHITE)
        cat = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_in_area(wave, 'A4', 'B6')
        self.play(Create(wave))
        self.play(Transform(wave, dot.move_to(wave.get_center())))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        self.place_in_area(bell, 'C4', 'D6')
        self.play(Create(bell))
        spike = Line(start=bell.get_center() + DOWN*0.5, end=bell.get_center() + UP*0.5, color=YELLOW)
        self.play(Transform(bell, spike))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(RED))
        self.place_at_grid(probe, 'E5', scale_factor=0.5)
        self.play(FadeIn(probe))
        self.play(probe.animate.shift(UP * 0.5))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(WHITE))
        self.place_at_grid(box, 'A5', scale_factor=0.5)
        self.place_at_grid(cat, 'A5', scale_factor=0.2)
        cat.set_opacity(0.3)
        self.play(FadeIn(box), FadeIn(cat))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(WHITE))
        self.play(FadeOut(box), cat.animate.set_opacity(1.0).scale(0.5))
        self.wait(1)
