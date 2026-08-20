from manim import *

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
            "Bayes' Theorem reverses conditional probabilities.",
            "Formula connects P(A|B) and P(B|A).",
            "It updates prior beliefs with new evidence.",
            "Prior probability evolves into posterior probability.",
            "Used to diagnose rare events accurately."
        ]
        self.setup_layout("Bayes' Theorem: The Logic of Reversal", lecture_lines)
        
        formula = MathTex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}", color=WHITE)
        stethoscope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/stethoscope.svg")
        prior_bar = Rectangle(height=0.5, width=2, color="#00FFFF", fill_opacity=0.5)
        evidence_bar = Rectangle(height=0.5, width=2, color="#00FFFF", fill_opacity=0.5)
        prior_label = Text("P(A)", font_size=20, color=WHITE)
        evidence_label = Text("P(B)", font_size=20, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.place_in_area(formula, 'A2', 'B5', scale_factor=1.0)
        self.place_at_grid(stethoscope, 'A6', scale_factor=0.3)
        self.play(Write(formula), FadeIn(stethoscope))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.place_at_grid(prior_bar, 'D2', scale_factor=0.9)
        self.place_at_grid(prior_label, 'D1', scale_factor=0.9)
        self.place_at_grid(evidence_bar, 'D5', scale_factor=0.9)
        self.place_at_grid(evidence_label, 'D6', scale_factor=0.9)
        self.play(Create(prior_bar), Write(prior_label), Create(evidence_bar), Write(evidence_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.play(Swap(prior_bar, evidence_bar))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        # Highlight P(B|A) in the formula. MathTex parts index carefully
        p_b_given_a = formula[0][2:7]
        self.play(Indicate(p_b_given_a, color="#FF00FF"))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.play(stethoscope.animate.scale(2).move_to(self.grid['E4']))
        self.wait(1)
