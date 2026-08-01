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

class Section7Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Add one to both sides to reach the final form.",
            "This formula links the most important constants in math.",
            "It merges geometry, algebra, and calculus into one.",
            "A simple expression reveals the universe's deep symmetry.",
            "It remains the crown jewel of mathematical beauty."
        ]
        
        self.setup_layout("Conclusion: The Identity Restored", lecture_lines)

        # Build the formula manually to avoid LaTeX dependency
        e = Text("e", font_size=48)
        ipi = Text("iπ", font_size=32).next_to(e.get_corner(UR), RIGHT, buff=0.05).shift(UP*0.1)
        term1 = VGroup(e, ipi)
        
        equals_neg_1 = Text(" = -1", font_size=48).next_to(term1, RIGHT)
        formula1 = VGroup(term1, equals_neg_1)
        
        # Area positioning for centered formula on right side
        self.place_in_area(formula1, "B2", "D6", scale_factor=1.2)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Write(formula1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(GRAY), self.lecture[1].animate.set_color("#90EE90"))
        
        plus_1_equals_0 = Text(" + 1 = 0", font_size=48, color="#90EE90").next_to(term1, RIGHT)
        
        # Animate the transition
        self.play(
            Transform(equals_neg_1, plus_1_equals_0),
            term1.animate.set_color("#90EE90")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(GRAY), self.lecture[2].animate.set_color("#FF69B4"))
        
        # Pulse colors - Text(" + 1 = 0") has characters '+', '1', '=', '0' at indices 0, 1, 2, 3
        self.play(Indicate(e, color=RED))
        self.play(Indicate(ipi[0], color=BLUE)) # i
        self.play(Indicate(ipi[1], color=GREEN)) # π
        self.play(Indicate(plus_1_equals_0[1], color=ORANGE)) # 1 (Index 1)
        self.play(Indicate(plus_1_equals_0[3], color=PURPLE)) # 0 (Index 3)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(GRAY), self.lecture[3].animate.set_color("#FFFFE0"))
        
        final_formula = VGroup(term1, equals_neg_1) # equals_neg_1 has been transformed to look like plus_1_equals_0
        self.play(
            final_formula.animate.scale(1.5).set_color("#FFFFE0")
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(GRAY), self.lecture[4].animate.set_color("#FFD700"))
        
        self.play(Indicate(final_formula, color="#FFD700", scale_factor=1.1))
        
        # Extra emphasis for the end
        surround_rect = SurroundingRectangle(final_formula, color="#FFD700", buff=0.5)
        self.play(Create(surround_rect))
        self.wait(3)
