from manim import *
import os

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
        lines = [
            "Bayes' Theorem: Reversing the logic of evidence.",
            "We calculate P(A|B) using the likelihood.",
            "It connects P(B|A) back to our original prior.",
            "This updates our belief with new data.",
            "Crucial for real-world smart device accuracy."
        ]
        self.setup_layout("Bayes' Theorem: The Logic of Reversing Evidence", lines)
        
        # === Animation for Lecture Line 1 ===
        # Fix: Using raw string (r"...") to avoid double-escaping issues with LaTeX backslashes
        formula = MathTex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}", font_size=48)
        # Fix 1: Properly center formula per VideoCritic
        self.place_in_area(formula, 'B2', 'C5', scale_factor=0.6)
        self.play(Write(formula))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        p_b_a = formula[0][8:13] # Isolating P(B|A)
        self.play(p_b_a.animate.set_color(YELLOW))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        arrow = Arrow(start=self.grid['C3'], end=self.grid['B3'], color=WHITE)
        self.play(Create(arrow))
        self.lecture[2].set_color(WHITE)

        # === Animation for Lecture Line 4 ===
        p_b = formula[0][16:19] # Isolating P(B)
        self.play(Indicate(p_b, color=PINK))
        self.lecture[3].set_color(PINK)

        # === Animation for Lecture Line 5 ===
        # Smartphone asset
        smartphone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/smartphone.svg")
        # Fix 2: Move scale icon / smartphone
        self.place_at_grid(smartphone, 'E5', scale_factor=0.7)
        self.play(FadeIn(smartphone))
        
        # Fix 3: Supplementary graphic
        supplementary_graphic = Circle(radius=0.5, color=GREEN)
        self.place_in_area(supplementary_graphic, 'D1', 'F2', scale_factor=0.5)
        self.play(FadeIn(supplementary_graphic))
        
        self.lecture[4].set_color(BLUE)
        
        self.wait(2)
