from manim import *
import numpy as np
import os

class Section4Scene(Scene):
    def construct(self):
        # Configuration and Directory Setup
        os.makedirs(os.path.join("media", "texts"), exist_ok=True)
        self.camera.background_color = "#000000"

        # 1. Title and Layout Initialization
        title_text = "Bayes' Theorem: The Geometry of Logic"
        lecture_lines = [
            "- Prior Belief: P(H)",
            "- New Evidence: E",
            "- Likelihood: P(E|H)",
            "- Posterior: P(H|E)"
        ]
        
        self.display_sidebar(title_text, lecture_lines)
        self.visualize_bayes_geometry()
        self.wait(2)

    def display_sidebar(self, title_text, lecture_lines):
        # Title at the top
        title = Text(title_text, font_size=32, color=WHITE).to_edge(UP, buff=0.5)
        self.add(title)

        # Left-side sidebar content
        lecture_mobjects = [Text(line, font_size=24, color=GRAY_A) for line in lecture_lines]
        sidebar = VGroup(*lecture_mobjects).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        sidebar.to_edge(LEFT, buff=0.7)
        
        self.play(Create(sidebar), run_time=1.5)
        self.play(sidebar[0].animate.set_color(BLUE), sidebar[3].animate.set_color(YELLOW))

    def visualize_bayes_geometry(self):
        # Define Coordinate Space for the Visualization (Right side of screen)
        # Grid Center is roughly at x=3
        base_box = Square(side_length=4, color=WHITE, stroke_width=2).shift(RIGHT * 3)
        base_label = Text("Sample Space", font_size=18).next_to(base_box, DOWN)

        # Area representing Prior P(H)
        prior_rect = Rectangle(
            width=2.0, height=4.0, 
            fill_opacity=0.4, 
            color=BLUE, 
            stroke_width=1
        ).align_to(base_box, LEFT).shift(RIGHT * 3)
        
        prior_label = Text("P(H)", font_size=20, color=BLUE).move_to(prior_rect.get_center())

        # Area representing Evidence E intersecting H
        evidence_rect = Rectangle(
            width=4.0, height=1.5, 
            fill_opacity=0.4, 
            color=GREEN, 
            stroke_width=1
        ).align_to(base_box, DOWN).shift(RIGHT * 3)
        
        evidence_label = Text("Evidence (E)", font_size=20, color=GREEN).next_to(evidence_rect, RIGHT, buff=0.1)

        # The intersection (Likelihood/Numerator)
        intersection = Intersection(prior_rect, evidence_rect, fill_opacity=0.8, color=YELLOW)
        
        # Animations
        self.play(Create(base_box), Write(base_label))
        self.wait(0.5)
        self.play(FadeIn(prior_rect), Write(prior_label))
        self.play(FadeIn(evidence_rect), Write(evidence_label))
        self.wait(1)
        
        self.play(
            intersection.animate.set_stroke(color=WHITE, width=2),
            Indicate(intersection)
        )
        
        formula = MathTex(
            "P(H|E) = \\frac{P(E|H)P(H)}{P(E)}", 
            font_size=36, 
            color=YELLOW
        ).shift(RIGHT * 3 + UP * 2.5)
        
        self.play(Write(formula))

    def get_grid_pos(self, row_idx, col_idx):
        """
        Helper to calculate grid positions for elements on the right side.
        Rows: 0-5, Cols: 0-5
        """
        x_start = 1.0
        y_start = 2.0
        return np.array([x_start + col_idx * 0.8, y_start - row_idx * 0.8, 0])