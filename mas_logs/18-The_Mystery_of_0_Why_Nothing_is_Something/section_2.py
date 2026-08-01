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
        lecture_lines = [
            "A factorial multiplies integers from n down to one.",
            "Three birds on a wire offer six possible arrangements.",
            "Factorials count the ways to arrange any set of items."
        ]
        
        self.setup_layout("Prerequisite: What is a Factorial?", lecture_lines)
        
        # Helper to create a bird asset using procedural shapes to avoid file errors
        def create_bird(color):
            body = Ellipse(width=0.6, height=0.4, color=color, fill_opacity=1)
            beak = Triangle(color=color, fill_opacity=1).scale(0.1).rotate(-PI/2).move_to(body.get_right())
            bird = VGroup(body, beak)
            return bird

        # === Animation for Lecture Line 1 ===
        # Center the text 'n! = n \times (n-1) \times \dots \times 1' (color #FFFFFF)
        self.play(self.lecture[0].animate.set_color(WHITE))
        formula = Text("n! = n × (n - 1) × ... × 1", color="#FFFFFF", font_size=24)
        self.place_in_area(formula, "B1", "B6", scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show 3 procedural birds (Red #FF0000, Blue #0000FF, Green #00FF00) sitting on a wire
        self.play(self.lecture[1].animate.set_color(WHITE))
        
        wire = Line(self.grid["D1"], self.grid["D6"], color=GRAY)
        bird_red = create_bird("#FF0000")
        bird_blue = create_bird("#0000FF")
        bird_green = create_bird("#00FF00")
        
        self.place_at_grid(bird_red, "D2", scale_factor=0.4)
        self.place_at_grid(bird_blue, "D4", scale_factor=0.4)
        self.place_at_grid(bird_green, "D6", scale_factor=0.4)

        # Calculation text above wire
        calc_3 = Text("3 × 2 × 1 = 6", font_size=24, color=WHITE)
        self.place_in_area(calc_3, "C1", "C6", scale_factor=1.0)

        self.play(Create(wire))
        self.play(FadeIn(bird_red, bird_blue, bird_green, shift=UP))
        self.play(Write(calc_3))
        
        # Swapping places to show arrangements
        self.play(
            bird_red.animate.move_to(self.grid["D4"]),
            bird_blue.animate.move_to(self.grid["D2"]),
            run_time=0.4
        )
        self.play(
            bird_red.animate.move_to(self.grid["D6"]),
            bird_green.animate.move_to(self.grid["D4"]),
            run_time=0.4
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fade out one bird, show 2 birds swapping (2 ways), then fade out another to show 1 bird (1 way)
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        
        # Transition to 2 birds
        self.play(FadeOut(bird_green), FadeOut(calc_3))
        calc_2 = Text("2 × 1 = 2", font_size=24, color=WHITE)
        self.place_in_area(calc_2, "C1", "C6", scale_factor=1.0)
        self.play(Write(calc_2))
        
        self.play(
            bird_red.animate.move_to(self.grid["D2"]),
            bird_blue.animate.move_to(self.grid["D4"]),
            run_time=0.4
        )
        self.play(
            bird_red.animate.move_to(self.grid["D4"]),
            bird_blue.animate.move_to(self.grid["D2"]),
            run_time=0.4
        )
        
        # Transition to 1 bird
        self.play(FadeOut(bird_blue), FadeOut(calc_2))
        calc_1 = Text("1 = 1", font_size=24, color=WHITE)
        self.place_in_area(calc_1, "C1", "C6", scale_factor=1.0)
        self.play(Write(calc_1))
        self.play(bird_red.animate.move_to(self.grid["D4"]))
        
        # Text 'n! = ways to arrange n items' (color #00FFFF)
        meaning_text = Text("n! = ways to arrange n items", font_size=24, color="#00FFFF")
        self.place_in_area(meaning_text, "E1", "E6", scale_factor=0.8)
        self.play(Write(meaning_text))
        
        self.wait(2)
