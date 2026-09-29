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
        lecture_lines = [
            "We use entropy for information gain.",
            "Formula: H(X) = -Σ p(x) log2 p(x).",
            "Aim for a balanced tree structure.",
            "Seek patterns with diverse outcomes.",
            "Higher entropy means better guesses."
        ]
        self.setup_layout("Mathematical Framework: Expected Information Gain", lecture_lines)
        
        # Assets
        dice_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg")
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        eq = MathTex("I(X) = -\\log_2(P(X))", color="#00FFFF")
        self.place_at_grid(eq, 'B4', scale_factor=1.0)
        self.place_at_grid(dice_icon, 'B3', scale_factor=0.3)
        self.play(Write(eq), FadeIn(dice_icon))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        highlight = Circle(radius=0.5, color=RED).set_stroke(width=2)
        self.place_at_grid(highlight, 'C4', scale_factor=0.6)
        self.play(Create(highlight))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(ORANGE))
        axes = Axes(x_range=[0, 4, 1], y_range=[0, 2, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: np.log2(x + 0.1), color=WHITE)
        graph = VGroup(axes, curve)
        self.place_in_area(graph, 'D3', 'F6', scale_factor=0.4)
        self.place_at_grid(magnifying_glass, 'D6', scale_factor=0.2)
        self.play(Create(graph), FadeIn(magnifying_glass))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(PURPLE))
        self.wait(1)
