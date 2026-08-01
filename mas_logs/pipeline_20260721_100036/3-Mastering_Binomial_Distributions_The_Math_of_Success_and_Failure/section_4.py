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
        # Data from storyboard
        title = "Deconstructing the Binomial Formula"
        lecture_lines = [
            "The binomial formula calculates the probability of success.",
            "Combination n-C-k counts the number of valid paths.",
            "p to the k represents the successes.",
            "q to the n-minus-k represents the failures.",
            "Multiply these parts to find the total probability."
        ]
        
        self.setup_layout(title, lecture_lines)
        
        # Hex Colors
        WHITE_C = "#FFFFFF"
        GREEN_C = "#90EE90"
        BLUE_C = "#ADD8E6"
        CORAL_C = "#F08080"
        HIGHLIGHT_C = "#FFFF00" 
        
        # === Animation for Lecture Line 1 ===
        # The binomial formula calculates the probability of success.
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_C))
        
        # Full formula: P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}
        formula = MathTex(
            "P(X=k)", "=", "{n \\choose k}", "p^k", "(1-p)^{n-k}",
            font_size=42, color=WHITE_C
        )
        # formula[0] = P(X=k), [1] = =, [2] = \binom{n}{k}, [3] = p^k, [4] = (1-p)^{n-k}
        self.place_in_area(formula, 'B1', 'C6', scale_factor=1.0)
        self.play(Write(formula))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Combination n-C-k counts the number of valid paths.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(GREEN_C),
            formula[2].animate.set_color(GREEN_C)
        )
        
        label_ways = Text("Number of Ways", font_size=20, color=GREEN_C)
        # Resolved Issue 31: Fixed position and scale for label_ways
        self.place_at_grid(label_ways, 'A4', scale_factor=0.6)
        
        arrow_ways = Arrow(
            start=label_ways.get_bottom(), 
            end=formula[2].get_top(), 
            color=GREEN_C, 
            buff=0.1
        )
        self.play(Create(arrow_ways), FadeIn(label_ways))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # p to the k represents the successes.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(BLUE_C),
            formula[3].animate.set_color(BLUE_C)
        )
        
        label_success = Text("Success Prob", font_size=20, color=BLUE_C)
        # Resolved Issue 32: Fixed position and scale for label_success
        self.place_at_grid(label_success, 'A5', scale_factor=0.6)
        
        arrow_success = Arrow(
            start=label_success.get_bottom(), 
            end=formula[3].get_top(), 
            color=BLUE_C, 
            buff=0.1
        )
        self.play(Create(arrow_success), FadeIn(label_success))
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        # q to the n-minus-k represents the failures.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(CORAL_C),
            formula[4].animate.set_color(CORAL_C)
        )
        
        label_failure = Text("Failure Prob", font_size=20, color=CORAL_C)
        # Resolved Issue 33: Fixed scale for label_failure
        self.place_at_grid(label_failure, 'A6', scale_factor=0.6)
        
        arrow_failure = Arrow(
            start=label_failure.get_bottom(), 
            end=formula[4].get_top(), 
            color=CORAL_C, 
            buff=0.1
        )
        self.play(Create(arrow_failure), FadeIn(label_failure))
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        # Multiply these parts to find the total probability.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT_C)
        )
        
        # Substitute values: n=10, k=7, p=0.7 into the formula
        sub_formula = MathTex(
            "P(X=7)", "=", "{10 \\choose 7}", "(0.7)^7", "(0.3)^3",
            font_size=42, color=WHITE_C
        )
        self.place_in_area(sub_formula, 'E1', 'F6', scale_factor=1.0)
        
        sub_formula[2].set_color(GREEN_C)
        sub_formula[3].set_color(BLUE_C)
        sub_formula[4].set_color(CORAL_C)
        
        self.play(FadeIn(sub_formula))
        self.wait(2)
        
        # Final state
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
