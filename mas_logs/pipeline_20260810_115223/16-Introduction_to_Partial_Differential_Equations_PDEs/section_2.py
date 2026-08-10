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
            "Partial derivatives isolate change by direction.",
            "Display operator notation L(u) = f.",
            "We analyze slopes North and East separately.",
            "Observe the operator in action.",
            "This isolates directional rate of change."
        ]
        self.setup_layout("Defining the Operators", lecture_lines)
        
        # Color definitions
        c1, c2, c3, c4, c5 = "#FFD700", "#00BFFF", "#32CD32", "#FF6347", "#FF00FF"
        
        # Load Assets
        hiker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hiker.svg")
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        mountain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(c1))
        op_text = MathTex(r"L(u) = f", font_size=48, color=c1)
        self.place_at_grid(op_text, "B2", scale_factor=1.0)
        self.play(Write(op_text))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(c2))
        self.place_in_area(hiker, 'B4', 'C6', scale_factor=0.6)
        self.play(FadeIn(hiker))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(c3))
        formula = MathTex(r"L(au+bv) = aL(u)+bL(v)", font_size=32, color=c3)
        self.place_at_grid(formula, 'B3', scale_factor=0.7)
        self.play(Write(formula))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(c4))
        self.place_in_area(compass, 'D4', 'E6', scale_factor=0.6)
        self.play(FadeIn(compass))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(c5))
        self.place_in_area(mountain, 'E2', 'F4', scale_factor=0.6)
        self.play(FadeIn(mountain))
        self.wait(2)
