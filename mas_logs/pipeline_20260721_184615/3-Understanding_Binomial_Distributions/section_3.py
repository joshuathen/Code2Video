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
        # Data for setup
        title_text = "The Mathematical Recipe (The Formula)"
        lecture_lines = [
            "The formula calculates the probability of exactly k successes.",
            "'n choose k' counts the different ways to succeed.",
            "'p' to the 'k' represents the successes' probability.",
            "'1 minus p' to the 'n-k' represents failures.",
            "Combine these parts to find the total probability."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # The formula calculates the probability of exactly k successes.
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        
        # P(X=k) = nCk * p^k * (1-p)^(n-k)
        formula = MathTex(
            r"P(X=k)", r"=", r"\binom{n}{k}", r"p^k", r"(1-p)^{n-k}",
            font_size=36
        )
        # Resolution for Issue 22: formula placement to avoid lecture notes
        self.place_in_area(formula, 'B3', 'D6', scale_factor=1.0)
        
        # Integration of Asset (Issue 18)
        recipe_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/recipe.svg")
        self.place_at_grid(recipe_icon, 'A5', scale_factor=0.6)
        
        self.play(FadeIn(formula), FadeIn(recipe_icon))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # 'n choose k' counts the different ways to succeed.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FF00FF")
        )
        # Color 'nCk' Magenta (#FF00FF) and label it 'Combinations'
        self.play(formula[2].animate.set_color("#FF00FF"))
        
        comb_label = Text("Combinations", font_size=20, color="#FF00FF")
        self.place_at_grid(comb_label, 'E3', scale_factor=0.8)
        
        self.play(Write(comb_label))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # 'p' to the 'k' represents the successes' probability.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FFFF")
        )
        # Color 'p^k' Cyan (#00FFFF) and label 'Success Prob'
        self.play(formula[3].animate.set_color("#00FFFF"))
        
        # Resolution for Issue 23: Alignment for success label
        success_label = Text("Success Prob", font_size=20, color="#00FFFF")
        self.place_at_grid(success_label, 'E4', scale_factor=0.8)
        
        self.play(Write(success_label))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # '1 minus p' to the 'n-k' represents failures.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color("#FFFF00")
        )
        # Color '(1-p)^(n-k)' Yellow (#FFFF00) and label 'Failure Prob'
        self.play(formula[4].animate.set_color("#FFFF00"))
        
        # Resolution for Issue 24: Alignment for failure label
        failure_label = Text("Failure Prob", font_size=20, color="#FFFF00")
        self.place_at_grid(failure_label, 'E5', scale_factor=0.8)
        
        self.play(Write(failure_label))
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # Combine these parts to find the total probability.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color("#FFFFFF")
        )
        
        # Morph variables into values: 10C7 * 0.7^7 * 0.3^3
        calc_formula = MathTex(
            r"P(X=7)", r"=", r"\binom{10}{7}", r"0.7^7", r"0.3^3",
            font_size=36
        )
        # Match colors of the parts to the variables they replaced
        calc_formula[2].set_color("#FF00FF")
        calc_formula[3].set_color("#00FFFF")
        calc_formula[4].set_color("#FFFF00")
        
        # Keep consistent positioning
        self.place_in_area(calc_formula, 'B3', 'D6', scale_factor=1.0)
        
        self.play(
            ReplacementTransform(formula, calc_formula),
            FadeOut(comb_label),
            FadeOut(success_label),
            FadeOut(failure_label)
        )
        self.wait(2.0)
