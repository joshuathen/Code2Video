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
            "Fourier coefficients act as volume knobs for harmonics.",
            "Integration extracts the weight of each specific frequency.",
            "Orthogonality ensures different waves do not interfere.",
            "Adjusting coefficients changes the resulting wave shape.",
            "Think of this like a signal mixing board."
        ]
        self.setup_layout("Calculating Coefficients (1:45-3:15)", lecture_lines)
        
        # Pre-build objects
        formula = MathTex(r"c_n = \int f(x) e^{-inx} dx", font_size=40, color=WHITE)
        self.place_in_area(formula, 'C2', 'D5', scale_factor=0.85)
        
        # Asset Loading
        slider = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/slider.svg")
        mixer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mixer.svg")

        # === Animation for Lecture Line 1 ===
        self.play(Write(formula))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Highlight f(x) and e^{-inx}
        self.play(formula.animate.set_color_by_tex("f(x)", "#FF0000").set_color_by_tex("e^{-inx}", "#FF0000"))
        self.lecture[1].set_color("#FF0000")

        # === Animation for Lecture Line 3 ===
        # Vector visualization (simple arrows)
        v1 = Arrow(start=ORIGIN, end=RIGHT*0.8, color="#00FF00")
        v2 = Arrow(start=ORIGIN, end=UP*0.8, color="#00FF00")
        vecs = VGroup(v1, v2)
        self.place_in_area(vecs, 'B1', 'B2', scale_factor=1.0)
        self.play(Create(vecs))
        self.lecture[2].set_color("#00FF00")

        # === Animation for Lecture Line 4 ===
        # Slide effect
        self.place_in_area(slider, 'E3', 'E4', scale_factor=0.6)
        self.play(FadeIn(slider), formula.animate.set_color("#FFFF00"))
        self.lecture[3].set_color("#FFFF00")

        # === Animation for Lecture Line 5 ===
        # Mixing board
        self.place_in_area(mixer, 'E5', 'F6', scale_factor=0.6)
        self.play(FadeIn(mixer))
        self.lecture[4].set_color("#FFD700")
        self.wait(1)
