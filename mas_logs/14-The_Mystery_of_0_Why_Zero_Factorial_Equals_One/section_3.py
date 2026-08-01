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
        # Initialize layout with updated lecture lines
        self.setup_layout("The Algebraic Proof", [
            'Any factorial equals n times its predecessor.', 
            'Let us substitute n with the number one.', 
            'Simplifying gives one equals one times zero factorial.', 
            'Isolating zero factorial shows it must be one.', 
            'This formula confirms zero factorial equals one logically.'
        ])
        
        # Colors
        EMERALD = "#50C878"
        GOLD = "#FFD700"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(EMERALD))
        
        # Write initial formula
        # Resolved Issue 34: Expanded area to 'B2' to 'E5'
        formula = Text("n! = n × (n - 1)!", color=EMERALD)
        self.place_in_area(formula, 'B2', 'E5', scale_factor=1.5)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(EMERALD))
        
        # Transform n to 1
        # Resolved Issue 34: Expanded area to 'B2' to 'E5'
        formula_sub = Text("1! = 1 × (1 - 1)!", color=EMERALD)
        self.place_in_area(formula_sub, 'B2', 'E5', scale_factor=1.5)
        
        self.play(Transform(formula, formula_sub))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(EMERALD))
        
        # Simplify parenthesis
        # Resolved Issue 35: Expanded area to 'B2' to 'E5'
        formula_simp = Text("1 = 1 × 0!", color=EMERALD)
        self.place_in_area(formula_simp, 'B2', 'E5', scale_factor=1.5)
        
        self.play(Transform(formula, formula_simp))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(EMERALD))
        
        # Isolate 0!
        # Resolved Issue 35: Expanded area to 'B2' to 'E5'
        formula_iso = Text("1 = 0!", color=EMERALD)
        self.place_in_area(formula_iso, 'B2', 'E5', scale_factor=1.5)
        self.play(Transform(formula, formula_iso))
        self.wait(0.5)
        
        # Flip to 0! = 1
        # Resolved Issue 36: Adjusted scale to 1.6 and expanded area to 'B2' to 'E5'
        formula_ordered = Text("0! = 1", color=EMERALD)
        self.place_in_area(formula_ordered, 'B2', 'E5', scale_factor=1.6)
        self.play(Transform(formula, formula_ordered))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(GOLD))
        
        # Surround with glowing gold box
        box = SurroundingRectangle(formula, color=GOLD, buff=0.3, stroke_width=4)
        glow = box.copy().set_stroke(width=8, opacity=0.3)
        
        self.play(Create(box), FadeIn(glow))
        self.play(Indicate(formula, color=GOLD))
        self.wait(3)
