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
        lecture_lines = [
            "Modular arithmetic helps categorize prime patterns.",
            "Dirichlet’s theorem predicts infinite prime occurrences.",
            "Arithmetic progressions reveal hidden prime paths."
        ]
        self.setup_layout("Dirichlet’s Theorem: The Rules of the Game", lecture_lines)
        
        # Assets
        calc_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        abacus_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/abacus.svg")
        
        # Elements
        formula = MathTex("a + nd").set_color("#00FF00")
        
        # 1. Modular arithmetic animation
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        
        # Integrating calculator asset with formula
        self.place_at_grid(formula, 'B3', scale_factor=0.7)
        self.place_at_grid(calc_icon, 'B1', scale_factor=0.5)
        self.play(Write(formula), FadeIn(calc_icon))
        self.wait(1)

        # 2. Dirichlet’s theorem animation
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        
        # Create grid of numbers
        numbers = VGroup(*[Text(str(i), font_size=20) for i in range(1, 13)])
        numbers.arrange_in_grid(2, 6, buff=0.5)
        self.place_in_area(numbers, 'C3', 'E4', scale_factor=0.9)
        self.play(FadeIn(numbers))
        self.wait(1)

        # 3. Arithmetic progressions reveal hidden prime paths
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF69B4"))
        
        # Highlight primes (2, 3, 5, 7, 11) with abacus icon nearby
        prime_indices = [1, 2, 4, 6, 10]
        highlights = VGroup(*[SurroundingRectangle(numbers[i], color="#FF69B4") for i in prime_indices])
        
        self.place_at_grid(abacus_icon, 'F3', scale_factor=0.5)
        self.play(Create(highlights), FadeIn(abacus_icon))
        self.wait(2)
