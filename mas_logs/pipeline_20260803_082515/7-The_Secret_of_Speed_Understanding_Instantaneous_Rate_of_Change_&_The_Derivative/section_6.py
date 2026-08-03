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

class Section6Scene(TeachingScene):
    def construct(self):
        # Section information
        title_text = "Defining the Derivative"
        lecture_lines = [
            "- We formalize this \"zooming in\" process with math.",
            "- The limit definition calculates slope as h vanishes.",
            "- This result is the derivative, denoted as f prime.",
            "- It provides a formula for slope at any moment.",
            "- Now we can calculate the cheetah's exact speedometer reading."
        ]
        
        # Initialize scene layout
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.lecture[0].set_color(YELLOW)
        
        # The white (#FFFFFF) formula f'(x) = lim (h->0) [f(x+h) - f(x)] / h appears.
        self.formula = MathTex(
            "f'(x)", 
            "=", 
            "\\lim_{h \\to 0}", 
            "{f(x+h) - f(x) \\over h}",
            color=WHITE
        )
        # Fix from Issue 31: Use 'C1' to 'D6' and scale 1.0
        self.place_in_area(self.formula, 'C1', 'D6', scale_factor=1.0)
        
        self.play(Write(self.formula))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Update lecture highlight
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # The 'f(x+h) - f(x)' part of the formula glows yellow (#FFFF00).
        # In a fraction {A \over B}, sub-mobject 0 is A, sub-mobject 1 is the line, sub-mobject 2 is B.
        self.play(self.formula[3][0].animate.set_color("#FFFF00"), run_time=1.5)
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Update lecture highlight
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # The 'h' in the denominator glows cyan (#00FFFF).
        self.play(self.formula[3][2].animate.set_color("#00FFFF"), run_time=1.5)
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # Update lecture highlight
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Green (#00FF00) text 'Instantaneous Rate' appears above the equal sign.
        self.rate_label = Text("Instantaneous Rate", font_size=24, color="#00FF00")
        self.rate_label.next_to(self.formula[1], UP, buff=0.4)
        
        self.play(FadeIn(self.rate_label))
        self.wait(1.5)

        # === Animation for Lecture Line 5 ===
        # Update lecture highlight
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # The formula transitions into f'(x) = 2x in white (#FFFFFF) text.
        self.final_formula = MathTex("f'(x) = 2x", color=WHITE)
        # Fix from Issue 32: Place at 'C4' with scale 1.2
        self.place_at_grid(self.final_formula, 'C4', scale_factor=1.2)
        
        self.play(
            ReplacementTransform(self.formula, self.final_formula),
            FadeOut(self.rate_label),
            run_time=2
        )
        self.wait(3)
        
        # Reset color of the last line
        self.lecture[4].set_color(WHITE)
        self.wait(1)
