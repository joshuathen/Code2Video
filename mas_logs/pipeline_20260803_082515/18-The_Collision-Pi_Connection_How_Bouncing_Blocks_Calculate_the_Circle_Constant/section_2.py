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
        title_text = "The Pattern: A Mysterious Sequence"
        lecture_lines = [
            "Equal masses result in exactly three total collisions.",
            "Increase the large mass by one hundred times.",
            "Now we count thirty-one collisions between the blocks.",
            "At ten thousand times the mass, it's three-hundred-fourteen.",
            "The digits of Pi appear as mass ratios grow."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Assets
        blocks_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/blocks.svg"

        # Column Labels (Issues 39 & 40)
        ratio_label = Text("Mass Ratio", font_size=24, color=WHITE)
        hits_label = Text("Hits", font_size=24, color=WHITE)
        self.place_at_grid(ratio_label, "A3", scale_factor=0.8)
        self.place_at_grid(hits_label, "A5", scale_factor=0.8)

        # Row 1: 1:1
        ratio_1 = MathTex("1 : 1", font_size=36, color="#ADD8E6") # Light Blue
        blocks_1 = SVGMobject(blocks_path, color="#ADD8E6")
        hits_1 = MathTex("3", font_size=36, color=WHITE)
        self.place_at_grid(ratio_1, "B3")
        self.place_at_grid(blocks_1, "B4", scale_factor=0.6)
        self.place_at_grid(hits_1, "B5")

        # Row 2: 1:100
        ratio_100 = MathTex("1 : 100", font_size=36, color="#ADD8E6")
        blocks_100 = SVGMobject(blocks_path, color="#ADD8E6")
        hits_31 = MathTex("31", font_size=36, color=WHITE)
        self.place_at_grid(ratio_100, "C3")
        self.place_at_grid(blocks_100, "C4", scale_factor=0.6)
        self.place_at_grid(hits_31, "C5")

        # Row 3: 1:10,000
        ratio_10k = MathTex("1 : 10,000", font_size=36, color="#ADD8E6")
        blocks_10k = SVGMobject(blocks_path, color="#ADD8E6")
        hits_314 = MathTex("314", font_size=36, color=WHITE)
        self.place_at_grid(ratio_10k, "D3")
        self.place_at_grid(blocks_10k, "D4", scale_factor=0.6)
        self.place_at_grid(hits_314, "D5")

        # Pi Symbol
        pi_symbol = MathTex(r"\pi", font_size=120, color="#00FFFF") # Cyan
        self.place_in_area(pi_symbol, "B6", "D6")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Write(ratio_label), Write(hits_label))
        self.play(FadeIn(ratio_1), Create(blocks_1), FadeIn(hits_1))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(FadeIn(ratio_100), Create(blocks_100))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(FadeIn(hits_31))
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(FadeIn(ratio_10k), Create(blocks_10k), FadeIn(hits_314))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Highlight digits 3, 1, 4 in yellow (#FFFF00)
        # Note: MathTex objects can be indexed by character, but here they are short.
        # hits_1 is just "3", hits_31 is "31", hits_314 is "314".
        self.play(
            hits_1.animate.set_color("#FFFF00"),
            hits_31.animate.set_color("#FFFF00"),
            hits_314.animate.set_color("#FFFF00")
        )
        self.play(FadeIn(pi_symbol))
        self.wait(3)

        # Reset last lecture line color
        self.lecture[4].set_color(WHITE)
        self.wait(1)
