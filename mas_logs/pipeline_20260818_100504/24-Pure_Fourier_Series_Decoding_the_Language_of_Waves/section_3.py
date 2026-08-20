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
            "Orthogonality isolates specific wave components perfectly.",
            "The integral acts as a mathematical sieve.",
            "We extract unique frequencies from chaotic mixtures.",
            "Each coefficient reveals a specific harmonic layer.",
            "Inner products filter signals into pure components."
        ]
        
        self.setup_layout("Calculating the Coefficients: The Inner Product", lecture_lines)
        
        # Assets
        wave_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wave.svg")
        wave1 = wave_svg.copy().set_color(BLUE)
        wave2 = wave_svg.copy().set_color(YELLOW)
        product_wave = wave_svg.copy().set_color(RED)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        wave_group = VGroup(wave1, wave2).arrange(RIGHT, buff=0.5)
        self.place_in_area(wave_group, "B2", "B4", scale_factor=0.6)
        self.play(Create(wave1), Create(wave2))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        sieve = Square(color=WHITE).scale(0.8)
        self.place_at_grid(sieve, "C3", scale_factor=1.0)
        self.play(Create(sieve))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.play(
            wave1.animate.move_to(self.grid["D2"]),
            wave2.animate.move_to(self.grid["D4"])
        )
        self.play(TransformFromCopy(wave1, product_wave), TransformFromCopy(wave2, product_wave))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(GREEN))
        coeff_text = Text("a_n", color=GREEN, font_size=36)
        self.place_at_grid(coeff_text, "D3", scale_factor=0.8)
        self.play(Write(coeff_text))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        bracket = Brace(VGroup(wave1, wave2), DOWN, color=PURPLE)
        self.play(Create(bracket))
        self.wait(2)
