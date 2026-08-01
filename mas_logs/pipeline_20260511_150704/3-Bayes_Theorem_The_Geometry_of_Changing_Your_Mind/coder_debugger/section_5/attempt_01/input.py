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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines
        lecture_lines = [
            "These geometric steps form the famous Bayes' Theorem formula.",
            "The gold term represents our initial prior belief.",
            "The yellow term is the likelihood from our evidence.",
            "We divide by the total probability of the evidence.",
            "This formula elegantly captures how we update our knowledge."
        ]
        self.setup_layout("The Bayesian Formula: From Shapes to Algebra", lecture_lines)
        
        # Define colors as requested
        prior_gold = "#FFD700"
        likelihood_yellow = "#FFFACD"
        evidence_white = "#FFFFFF"

        # Create geometric visual areas (initial state)
        # prior_box: represents P(A)
        prior_box = Rectangle(width=0.5, height=2.0, color=prior_gold, fill_opacity=0.6)
        # likelihood_circle: represents P(B|A) (the constraint)
        likelihood_circle = Circle(radius=0.4, color=likelihood_yellow, fill_opacity=0.6)
        # evidence_tri: represents P(B) (the total evidence)
        evidence_tri = Triangle().scale(0.5).set_color(evidence_white).set_fill(evidence_white, opacity=0.3)

        # Place initial objects according to updated feedback (Issues 33, 34, 35)
        self.place_at_grid(prior_box, "B2")
        self.place_at_grid(likelihood_circle, "D4")
        self.place_at_grid(evidence_tri, "F6")

        # Define the MathTex formula parts for modular transformation/coloring
        # P(A|B) = [P(B|A) P(A)] / P(B)
        formula = MathTex(
            "P(A|B)", "=", "{ P(B|A)", "P(A)", "\\over", "P(B) }",
            font_size=42
        )
        # Map formula indices for clarity: 0:P(A|B), 1:=, 2:P(B|A), 3:P(A), 4:fraction line, 5:P(B)
        self.place_in_area(formula, "B3", "E6")

        # === Animation for Lecture Line 1 ===
        # These geometric steps form the famous Bayes' Theorem formula.
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(
            FadeIn(prior_box),
            FadeIn(likelihood_circle),
            FadeIn(evidence_tri)
        )
        self.wait(1)

        self.play(
            ReplacementTransform(prior_box, formula[3]),
            ReplacementTransform(likelihood_circle, formula[2]),
            ReplacementTransform(evidence_tri, formula[5]),
            Write(formula[0]),
            Write(formula[1]),
            Write(formula[4]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The gold term represents our initial prior belief.
        self.play(self.lecture[1].animate.set_color(prior_gold))
        self.play(
            formula[3].animate.set_color(prior_gold),
            formula[3].animate.scale(1.2).set_rate_func(there_and_back),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The yellow term is the likelihood from our evidence.
        self.play(self.lecture[2].animate.set_color(likelihood_yellow))
        self.play(
            formula[2].animate.set_color(likelihood_yellow),
            formula[2].animate.scale(1.2).set_rate_func(there_and_back),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # We divide by the total probability of the evidence.
        self.play(self.lecture[3].animate.set_color(evidence_white))
        underline = Underline(formula[5], color=evidence_white, buff=0.1)
        self.play(
            formula[5].animate.set_color(evidence_white),
            Create(underline),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This formula elegantly captures how we update our knowledge.
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        # Pulsing glow animation
        glow = formula.copy().set_stroke(WHITE, width=8).set_opacity(0.3)
        self.add(glow)
        self.play(
            glow.animate.scale(1.1).set_opacity(0),
            formula.animate.scale(1.05),
            rate_func=there_and_back,
            run_time=2
        )
        self.remove(glow)
        self.wait(2)
