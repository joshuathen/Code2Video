from manim import *
import numpy as np

# Use the base class provided in the prompt
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
        # Setup Content
        lecture_lines = [
            'The formula calculates the probability of exactly k successes.',
            'n-choose-k counts all possible paths to reach that outcome.',
            'p to the k represents the probability of successes.',
            'One-minus-p handles the remaining failure outcomes.',
            'Together, they define the probability of a specific result.'
        ]
        self.setup_layout("Visualizing the Formula: Combinations and Probability", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # The formula calculates the probability of exactly k successes.
        # Replacing MathTex with Text to avoid FileNotFoundError: 'latex'
        prob_stat = Text("P(X=k)", font_size=36)
        self.place_at_grid(prob_stat, 'B3', scale_factor=1.1)

        n_choose_k = Text("n-choose-k", font_size=30, color="#FFFF00")
        pk = Text("p^k", font_size=36, color="#00FF00")
        success_terms = VGroup(n_choose_k, pk).arrange(RIGHT, buff=0.4)
        self.place_in_area(success_terms, 'C2', 'C4', scale_factor=1.0)

        failure_term = Text("(1-p)^(n-k)", font_size=30, color="#FF0000")
        self.place_in_area(failure_term, 'C5', 'C6', scale_factor=0.85)

        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(FadeIn(prob_stat), FadeIn(success_terms), FadeIn(failure_term))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # n-choose-k counts all possible paths to reach that outcome.
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Galton Board pegs representing choices
        peg_coords = ['D3', 'D4', 'E2', 'E3', 'E4', 'E5', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6']
        pegs = VGroup(*[Dot(self.grid[pos], radius=0.08, color=WHITE) for pos in peg_coords])
        
        # Path showing a specific route to 2 successes (k=2) in 3 trials
        path_y = VGroup(
            Line(self.grid['C3'], self.grid['D3'], color="#FFFF00"),
            Line(self.grid['D3'], self.grid['E4'], color="#FFFF00"),
            Line(self.grid['E4'], self.grid['F4'], color="#FFFF00")
        )
        
        self.play(Create(pegs))
        self.play(Create(path_y))
        self.play(Flash(n_choose_k, color="#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # p to the k represents the probability of successes.
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        ball = Dot(self.grid['D3'], color=WHITE)
        success_segment = Line(self.grid['D3'], self.grid['E4'], color="#00FF00", stroke_width=5)
        
        self.play(Indicate(pk, color="#00FF00"))
        self.add(ball)
        self.play(ball.animate.move_to(self.grid['E4']), Create(success_segment))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # One-minus-p handles the remaining failure outcomes.
        self.play(self.lecture[3].animate.set_color("#FF0000"))
        
        failure_segment = Line(self.grid['E4'], self.grid['F3'], color="#FF0000", stroke_width=5)
        
        self.play(Indicate(failure_term, color="#FF0000"))
        self.play(ball.animate.move_to(self.grid['F3']), Create(failure_segment))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Together, they define the probability of a specific result.
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        all_terms = VGroup(prob_stat, success_terms, failure_term)
        self.play(Indicate(all_terms, color=WHITE))
        self.wait(2)
