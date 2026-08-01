from manim import *
import numpy as np

class Section4Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # Base setup
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP, buff=0.5)
        self.add(self.title)

        # Left-side lecture content
        lecture_texts = [Text(line, font_size=20, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.7)
        self.add(self.lecture)

        # Define fine-grained animation grid (6x6 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]
        cols = ["1", "2", "3", "4", "5", "6"]

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Center the grid on the right half of the screen
                x = 1.5 + j * 0.9
                y = 2.0 - i * 0.8
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=1.0):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def construct(self):
        # Initialization of the lesson
        title_str = "The Riemann Zeta Function: DNA of Primes"
        lines = [
            "- The Zeta Function Equation",
            "- The Euler Product Formula",
            "- Convergence and Analytic Continuity",
            "- Encoding Prime Distributions"
        ]
        
        self.setup_layout(title_str, lines)
        
        # 1. Zeta Function Representation
        # Changed from MathTex to Text to bypass missing 'latex' dependency
        zeta_formula = Text(
            "ζ(s) = Σ 1/n^s", 
            color=BLUE
        )
        self.place_at_grid(zeta_formula, "B3", scale_factor=1.1)
        
        # 2. Euler Product Representation (The "DNA" link)
        # Changed from MathTex to Text to bypass missing 'latex' dependency
        euler_product = Text(
            "ζ(s) = Π 1/(1 - p^-s)",
            color=YELLOW
        )
        self.place_at_grid(euler_product, "D3", scale_factor=1.1)
        
        # 3. Visualizing the Link
        prime_list = VGroup(*[
            Text(p, font_size=24, color=GREEN) for p in ["2", "3", "5", "7", "11", "..."]
        ]).arrange(RIGHT, buff=0.4)
        self.place_at_grid(prime_list, "F3", scale_factor=1.0)
        
        # Animation sequence
        self.play(Write(zeta_formula))
        self.wait(1)
        
        # Show bridge arrow
        arrow = Arrow(
            start=zeta_formula.get_bottom(),
            end=euler_product.get_top(),
            color=WHITE,
            buff=0.2
        )
        
        self.play(Create(arrow))
        self.play(FadeIn(euler_product, shift=DOWN))
        self.wait(1)
        
        # Linking to primes
        self.play(Write(prime_list))
        
        # Highlighting a specific term
        box = SurroundingRectangle(euler_product, color=RED, buff=0.1)
        self.play(Create(box))
        
        # Final emphasis
        conclusion = Text("Primes are the 'atoms' of Zeta", font_size=20, slant=ITALIC)
        self.place_at_grid(conclusion, "E5", scale_factor=0.8)
        self.play(Write(conclusion))
        
        self.wait(3)