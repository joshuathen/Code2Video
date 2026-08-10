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
        lines = [
            "Classical states are either here or there.",
            "Quantum states represent possibilities as a vector.",
            "We call this quantum state vector, |ψ>.",
            "Think of a coin blurring between heads and tails.",
            "Quantum reality exists as a multi-dimensional mathematical space."
        ]
        self.setup_layout("Prerequisite: The Classical vs. Quantum Divide", lines)
        
        # --- Mobjects ---
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg]
        classical_coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color="#FF0000")
        label_0 = Text("0", font_size=24, color="#FF0000")
        
        superposition_coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color="#00FF00")
        coeff_text = MathTex(r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle", font_size=30, color="#00FF00")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF0000"), FadeIn(self.place_at_grid(classical_coin, 'B3', 0.5)))
        self.play(FadeIn(self.place_at_grid(label_0, 'C3', 1.0)))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00FF00"), 
                  Transform(classical_coin, self.place_at_grid(superposition_coin, 'B3', 0.5)),
                  FadeOut(label_0))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"), Write(self.place_at_grid(coeff_text, 'D3', 0.7)))

        # === Animation for Lecture Line 4 ===
        glow = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg", color="#00FF00").scale(0.6).set_opacity(0.3)
        glow.move_to(self.grid['B3'])
        self.add(glow)
        self.play(self.lecture[3].animate.set_color("#FFFF00"), Indicate(classical_coin, scale_factor=1.2, color="#00FF00"))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF00FF"), FadeOut(glow))
        self.wait(1)
