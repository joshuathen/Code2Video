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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Visualizing Superposition", [
            "Quantum superposition allows multiple states simultaneously.",
            "Dirac notation describes this mathematical state.",
            "The Bloch Sphere visualizes these probabilities.",
            "Think of a spinning coin's blur.",
            "It exists as both heads and tails."
        ])
        
        # === Animation for Lecture Line 1 ===
        # State vector |0> and |1>
        vec0 = Arrow(start=ORIGIN, end=UP*1.5, color="#00FFFF")
        vec1 = Arrow(start=ORIGIN, end=DOWN*1.5, color="#FFFF00")
        label0 = MathTex(r"|0\rangle", color="#00FFFF").next_to(vec0, UP)
        label1 = MathTex(r"|1\rangle", color="#FFFF00").next_to(vec1, DOWN)
        
        group = VGroup(vec0, vec1, label0, label1)
        self.place_in_area(group, "A2", "F5", scale_factor=0.6)
        self.play(Create(vec0), Write(label0), Create(vec1), Write(label1))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        eqn = MathTex(r"|\psi\rangle = \alpha|0\rangle + \beta|1\rangle", color="#FFFFFF")
        # Fixed issue 25/37 (line 70)
        self.place_at_grid(eqn, "B2", scale_factor=0.7)
        self.play(Write(eqn))
        self.lecture[1].set_color("#FFFFFF")

        # === Animation for Lecture Line 3 ===
        sphere = Sphere(radius=1.5, fill_opacity=0.2, color=BLUE).set_stroke(width=1)
        # Fixed issue 24/37 (line 76)
        self.place_in_area(sphere, "C4", "E6", scale_factor=0.6)
        self.play(Create(sphere))
        self.lecture[2].set_color("#0000FF")

        # === Animation for Lecture Line 4 ===
        # Using asset [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg]
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        self.place_at_grid(coin, "D3", scale_factor=0.8)
        self.play(FadeIn(coin))
        self.play(Rotate(coin, angle=2*PI, run_time=2))
        self.lecture[3].set_color("#FF00FF")

        # === Animation for Lecture Line 5 ===
        final_state = MathTex(r"\text{Superposition}", color="#FF00FF")
        # Fixed issue 26/37 (line 89)
        self.place_at_grid(final_state, "F2", scale_factor=0.9)
        self.play(Write(final_state))
        self.lecture[4].set_color("#FF00FF")
