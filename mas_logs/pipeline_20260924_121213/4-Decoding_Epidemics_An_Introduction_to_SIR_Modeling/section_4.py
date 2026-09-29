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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Threshold of Chaos: Understanding R0", [
            "- The Basic Reproduction Number, R0, defines growth.",
            "- If R0 exceeds one, the epidemic grows.",
            "- If R0 is below one, it disappears."
        ])
        
        # Assets
        virus = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/virus.svg")
        hospital = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hospital.svg")
        
        formula = MathTex(r"R_0 = \frac{\beta}{\gamma}", font_size=48, color=WHITE)
        self.place_in_area(formula, 'B4', 'B5', scale_factor=1.2)
        
        r0_gt_1 = Text("R0 > 1", font_size=36, color=RED)
        self.place_at_grid(r0_gt_1, 'C3', scale_factor=1.0)
        
        r0_lt_1 = Text("R0 < 1", font_size=36, color=GREEN)
        self.place_at_grid(r0_lt_1, 'C5', scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        # Display R0 = beta/gamma formula in #FFFFFF [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/virus.svg].
        self.play(Write(formula), FadeIn(self.place_at_grid(virus, 'B2', scale_factor=0.6)))
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Flash R0 > 1 in #FF0000 color.
        self.play(Indicate(r0_gt_1, color=RED))
        self.play(self.lecture[1].animate.set_color(RED))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Simulate curve flattening at R0 < 1 in #00FF00 [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/hospital.svg].
        self.play(Indicate(r0_lt_1, color=GREEN), FadeIn(self.place_at_grid(hospital, 'D5', scale_factor=0.6)))
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.wait(1)
