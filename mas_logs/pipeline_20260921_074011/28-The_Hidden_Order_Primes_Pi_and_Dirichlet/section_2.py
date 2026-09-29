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
        self.setup_layout("The Pi Connection: Why do Primes and Pi converge?", [
            "Primes and Pi share a connection.",
            "Consider the probability of coprime integers.",
            "This probability equals 6 divided by Pi squared.",
            "Random number pairs exhibit this ratio.",
            "Global patterns emerge from local randomness."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/abacus.svg]
        abacus = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/abacus.svg")
        eq = MathTex(r"\prod_{p} \left(1 - \frac{1}{p^2}\right) = \frac{1}{\zeta(2)} = \frac{6}{\pi^2}", color="#FFD700")
        group_1 = VGroup(eq, abacus).arrange(DOWN, buff=0.5)
        self.place_in_area(group_1, 'A1', 'B6', scale_factor=0.9)
        self.play(Write(eq), FadeIn(abacus))
        self.lecture[0].set_color("#FFD700")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg]
        dice = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg")
        self.place_at_grid(dice, 'C2', scale_factor=0.5)
        dot_area = VGroup()
        for i in range(20):
            dot = Dot(color=WHITE, radius=0.05)
            dot.move_to(self.grid["C1"] + np.array([np.random.rand()*3, np.random.rand()*2, 0]))
            dot_area.add(dot)
        self.play(FadeIn(dice), FadeIn(dot_area))
        self.lecture[1].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#32CD32")
        for dot in dot_area:
            if np.random.rand() < 0.607:
                dot.set_color("#32CD32")
        self.play(*[dot.animate.set_color(dot.color) for dot in dot_area])
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        ratio = DecimalNumber(0.607, color="#FF4500", num_decimal_places=3)
        label = MathTex(r"\text{Ratio} \approx ", color="#FF4500")
        ratio_group = VGroup(label, ratio).arrange(RIGHT)
        self.place_at_grid(ratio_group, 'E2', scale_factor=0.8)
        self.play(Write(ratio_group))
        self.lecture[3].set_color("#FF4500")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg]
        calc = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        self.place_at_grid(calc, 'E6', scale_factor=0.4)
        axes = Axes(x_range=[0, 20], y_range=[0, 1], axis_config={"include_tip": False}, x_length=3, y_length=2)
        self.place_at_grid(axes, 'E5', scale_factor=0.7)
        self.play(FadeIn(calc), Create(axes))
        self.lecture[4].set_color("#FFFFFF")
        self.wait(1)
