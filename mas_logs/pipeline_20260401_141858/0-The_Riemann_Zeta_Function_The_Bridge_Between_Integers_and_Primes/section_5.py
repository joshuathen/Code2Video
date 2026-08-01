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

class Section5Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=32, color=WHITE).to_edge(UP, buff=0.5)
        self.add(self.title)

        # Left-side lecture content
        lecture_texts = []
        for line in lecture_lines:
            t = Text(line, font_size=20, color=WHITE)
            lecture_texts.append(t)
        
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        self.lecture.scale(0.9).to_edge(LEFT, buff=0.5)
        self.add(self.lecture)

        # Coordinate grid for visual elements on the right half
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 1.0 + j * 0.8
                y = 2.0 - i * 0.8
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def construct(self):
        # 1. Initialize Layout
        self.setup_layout(
            "The Euler Product Formula", 
            [
                "- The Analytic Bridge",
                "- Identity: Sum = Product",
                "- Mapping Integers to Primes",
                "- Convergence for Re(s) > 1"
            ]
        )

        # 2. Main Formula
        # Replaced MathTex with Text using Unicode to bypass missing LaTeX dependency
        # Zeta(s) = Product over primes p of (1 - p^-s)^-1
        zeta_formula = Text(
            "ζ(s) = Πₚ (1 - p⁻ˢ)⁻¹",
            font_size=36,
            color=YELLOW
        )
        self.place_at_grid(zeta_formula, "B3", scale_factor=1.1)

        self.play(Write(self.title))
        self.play(FadeIn(self.lecture, shift=RIGHT))
        self.play(Write(zeta_formula))
        self.wait(1)

        # 3. Example Expansion for Primes
        # Replaced MathTex with Text using Unicode to bypass missing LaTeX dependency
        expansion = Text(
            "= (1-2⁻ˢ)⁻¹ · (1-3⁻ˢ)⁻¹ · (1-5⁻ˢ)⁻¹ ...",
            font_size=24,
            color=BLUE_B
        )
        self.place_at_grid(expansion, "D3", scale_factor=0.9)
        
        self.play(FadeIn(expansion, shift=UP))
        self.wait(1)

        # 4. Highlight Connection
        highlight_box = SurroundingRectangle(zeta_formula, color=GOLD, buff=0.2)
        connection_text = Text("Primes", font_size=24, color=GOLD)
        connection_text.next_to(highlight_box, DOWN)

        self.play(Create(highlight_box))
        self.play(Write(connection_text))
        self.wait(2)

        # 5. Final Transition
        self.play(
            FadeOut(zeta_formula),
            FadeOut(expansion),
            FadeOut(highlight_box),
            FadeOut(connection_text)
        )
        
        conclusion = Text("The Foundation of Prime Number Theory", font_size=28, color=WHITE)
        self.place_at_grid(conclusion, "C3", scale_factor=0.8)
        self.play(Write(conclusion))
        self.wait(2)
