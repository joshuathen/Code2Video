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
        # 1. Setup Layout
        lecture_lines = [
            "Bayes' Theorem updates our beliefs with new evidence.",
            'Start with the prior: the initial chance of success.',
            'Add the likelihood: how well evidence supports the claim.',
            'Divide by total probability to normalize the result.',
            'This formula calculates the final, updated posterior probability.'
        ]
        self.setup_layout("The Core: Visualizing Bayes' Theorem", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Fixed: Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        # Bayes' Formula P(A|B) = [P(B|A) * P(A)] / P(B)
        bayes_formula = Text(
            "P(A|B) = [P(B|A) * P(A)] / P(B)",
            color="#ECF0F1",
            font_size=24
        )
        # Position formula in area A1 to B6
        self.place_in_area(bayes_formula, 'A1', 'B6', scale_factor=0.9)
        
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Write(bayes_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Prior: Forest square and gold strip
        forest_size = 3.0
        forest = Rectangle(
            width=forest_size, height=forest_size,
            fill_color="#2ECC71", fill_opacity=1.0, stroke_color=WHITE, stroke_width=1
        )
        
        # 1% Prior strip (using 0.05 for visibility in render)
        prior_width = forest_size * 0.05 
        prior_rect = Rectangle(
            width=prior_width, height=forest_size,
            fill_color="#F1C40F", fill_opacity=1.0, stroke_width=0
        ).align_to(forest, LEFT)
        
        # Visual Group for Area Alignment
        tp_rect = Rectangle(
            width=prior_width, height=forest_size * 0.9,
            fill_color="#E74C3C", fill_opacity=1.0, stroke_width=0
        ).align_to(prior_rect, UP).align_to(prior_rect, LEFT)
        
        fp_rect = Rectangle(
            width=forest_size - prior_width, height=forest_size * 0.1,
            fill_color="#E74C3C", fill_opacity=1.0, stroke_width=0
        ).align_to(forest, RIGHT).align_to(forest, DOWN)

        posterior_visual = VGroup(forest, prior_rect, tp_rect, fp_rect)
        # Position visual area in D2 to F5
        self.place_in_area(posterior_visual, 'D2', 'F5', scale_factor=0.8)

        prior_label = Text("Prior", font_size=24, color="#F1C40F")
        self.place_at_grid(prior_label, 'C2', scale_factor=0.8)
        
        forest_desc = Text("Sample Space: 100 Tiles", font_size=16, color=WHITE)
        forest_desc.next_to(forest, UP, buff=0.1)

        self.play(self.lecture[1].animate.set_color("#F1C40F"))
        self.play(Create(forest), Write(forest_desc))
        self.play(FadeIn(prior_rect), Write(prior_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Likelihood: highlight area in red
        likelihood_label = Text("Likelihood", font_size=24, color="#E74C3C")
        self.place_at_grid(likelihood_label, 'C4', scale_factor=0.8)

        self.play(self.lecture[2].animate.set_color("#E74C3C"))
        self.play(FadeIn(tp_rect), Write(likelihood_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Divide by total probability
        self.play(self.lecture[3].animate.set_color(WHITE))
        self.play(FadeIn(fp_rect))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Final updated posterior probability
        posterior_label = Text("Posterior", font_size=24, color="#ECF0F1")
        self.place_at_grid(posterior_label, 'F1', scale_factor=0.8)

        self.play(self.lecture[4].animate.set_color("#ECF0F1"))
        self.play(
            bayes_formula.animate.set_color("#ECF0F1"),
            Write(posterior_label)
        )
        self.wait(2)
