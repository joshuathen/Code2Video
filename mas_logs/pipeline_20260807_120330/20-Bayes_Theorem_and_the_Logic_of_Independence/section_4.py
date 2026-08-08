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
        # Data from shared state
        title = "The Core: Bayes' Theorem Formula"
        lecture_lines = [
            "Bayes' Theorem lets us reverse conditional probabilities.",
            "Start with your prior belief in the hypothesis.",
            "Multiply by the likelihood of seeing this evidence.",
            "Divide by the total probability of that evidence.",
            "Now you have the updated posterior probability."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Colors defined in requirements
        PRIOR_COLOR = "#FFFF00"
        LIKELIHOOD_COLOR = "#00FF00"
        EVIDENCE_COLOR = "#00FFFF"
        POSTERIOR_COLOR = "#FF00FF"
        HIGHLIGHT_COLOR = WHITE
        DIM_COLOR = GRAY

        # === Animation for Lecture Line 1 ===
        # Bayes' Theorem lets us reverse conditional probabilities.
        # Write the full Bayes formula P(A|B) = P(B|A)P(A)/P(B).
        
        # Formula parts: 0:P(A|B), 1:=, 2:{, 3:P(B|A), 4:P(A), 5:\over, 6:P(B), 7:}
        formula = MathTex(
            "P(A|B)", "=", "{", "P(B|A)", "P(A)", "\\over", "P(B)", "}"
        )
        
        # Place formula in the center of visual area (spanning B2 to D6 per Issue 41)
        self.place_in_area(formula, "B2", "D6", scale_factor=1.1)
        
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Start with your prior belief in the hypothesis.
        # Highlight P(A) in #FFFF00 and label it 'Prior'.
        prior_label = Text("Prior", font_size=24, color=PRIOR_COLOR)
        # P(A) is part 4 in the numerator
        self.place_at_grid(prior_label, "A5", scale_factor=0.8)
        
        self.play(
            self.lecture[0].animate.set_color(DIM_COLOR),
            self.lecture[1].animate.set_color(PRIOR_COLOR)
        )
        self.play(
            formula[4].animate.set_color(PRIOR_COLOR),
            FadeIn(prior_label, shift=UP*0.2)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Multiply by the likelihood of seeing this evidence.
        # Highlight P(B|A) in #00FF00 and label it 'Likelihood'.
        likelihood_label = Text("Likelihood", font_size=24, color=LIKELIHOOD_COLOR)
        # P(B|A) is part 3 in the numerator
        # Moved to A4 per Issue 42
        self.place_at_grid(likelihood_label, "A4", scale_factor=0.8)

        self.play(
            self.lecture[1].animate.set_color(DIM_COLOR),
            self.lecture[2].animate.set_color(LIKELIHOOD_COLOR)
        )
        self.play(
            formula[3].animate.set_color(LIKELIHOOD_COLOR),
            FadeIn(likelihood_label, shift=UP*0.2)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Divide by the total probability of that evidence.
        # Highlight P(B) in #00FFFF and label it 'Evidence'.
        evidence_label = Text("Evidence", font_size=24, color=EVIDENCE_COLOR)
        # P(B) is part 6 in the denominator
        # Moved to F4 per Issue 42
        self.place_at_grid(evidence_label, "F4", scale_factor=0.8)

        self.play(
            self.lecture[2].animate.set_color(DIM_COLOR),
            self.lecture[3].animate.set_color(EVIDENCE_COLOR)
        )
        self.play(
            formula[6].animate.set_color(EVIDENCE_COLOR),
            FadeIn(evidence_label, shift=DOWN*0.2)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Now you have the updated posterior probability.
        # Highlight P(A|B) in #FF00FF and label it 'Posterior'.
        posterior_label = Text("Posterior", font_size=24, color=POSTERIOR_COLOR)
        # P(A|B) is part 0 on the left side
        # Moved to A2 per Issue 40
        self.place_at_grid(posterior_label, "A2", scale_factor=0.8)

        self.play(
            self.lecture[3].animate.set_color(DIM_COLOR),
            self.lecture[4].animate.set_color(POSTERIOR_COLOR)
        )
        self.play(
            formula[0].animate.set_color(POSTERIOR_COLOR),
            FadeIn(posterior_label, shift=UP*0.2)
        )
        self.wait(3)
