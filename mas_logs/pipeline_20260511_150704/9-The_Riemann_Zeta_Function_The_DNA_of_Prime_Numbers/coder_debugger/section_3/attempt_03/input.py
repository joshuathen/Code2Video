from manim import *
import numpy as np
import pathlib

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        
        # FOCUSED FIX: Pre-emptively create the text directory to avoid FileExistsError race condition in Manim v0.19.0
        text_dir = pathlib.Path(config.get_dir("text_dir"))
        text_dir.mkdir(parents=True, exist_ok=True)
        
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
        # Setup layout
        lines = [
            'Leonhard Euler discovered a profound connection to prime numbers.', 
            'The infinite sum equals an infinite product of primes.', 
            'Each prime p contributes a unique factor to Zeta.', 
            'This formula bridges integers and the building blocks of math.', 
            'Zeta encodes the secret distribution of all prime numbers.'
        ]
        self.setup_layout("The Euler Product: The Bridge to Primes", lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        
        # FOCUSED FIX: Using MarkupText instead of MathTex to avoid FileNotFoundError: 'latex' 
        # in environments where LaTeX is not installed. MarkupText uses Pango (system-level).
        sum_tex = MarkupText("Σ n<sup>-s</sup>", color=WHITE)
        self.place_in_area(sum_tex, 'B1', 'C2', scale_factor=1.2)
        
        prod_tex = MarkupText("Π (1-p<sup>-s</sup>)<sup>-1</sup>", color=GREEN)
        self.place_in_area(prod_tex, 'B5', 'C6', scale_factor=1.2)
        
        self.play(Write(sum_tex))
        self.wait(0.5)
        self.play(Write(prod_tex))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(GREEN))
        
        expansion_parts = [
            "(1-2<sup>-s</sup>)<sup>-1</sup>",
            "(1-3<sup>-s</sup>)<sup>-1</sup>",
            "(1-5<sup>-s</sup>)<sup>-1</sup>",
            "(1-7<sup>-s</sup>)<sup>-1</sup>",
            "..."
        ]
        expansion = VGroup(*[MarkupText(part, color=GREEN, font_size=24) for part in expansion_parts]).arrange(RIGHT, buff=0.2)
        self.place_in_area(expansion, 'D2', 'D5', scale_factor=0.8)
        
        self.play(FadeIn(expansion, shift=UP))
        
        for i in range(4):
            self.play(Indicate(expansion[i], color=YELLOW, scale_factor=1.3), run_time=0.7)
            
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        
        bridge_arrow = DoubleArrow(
            sum_tex.get_right() + RIGHT*0.2, 
            prod_tex.get_left() + LEFT*0.2, 
            color="#FFFF00", 
            stroke_width=5
        )
        glow = bridge_arrow.copy().set_stroke(width=10, opacity=0.4)
        
        self.play(Create(bridge_arrow), FadeIn(glow))
        self.play(Indicate(bridge_arrow, color="#FFFF00"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(WHITE))
        
        final_zeta = MarkupText(
            "ζ(s) = Σ n<sup>-s</sup> = Π (1-p<sup>-s</sup>)<sup>-1</sup>",
            color=WHITE
        )
        self.place_in_area(final_zeta, 'C1', 'E6', scale_factor=1.1)
        
        self.play(
            FadeOut(expansion),
            FadeOut(bridge_arrow),
            FadeOut(glow),
            Transform(sum_tex, final_zeta),
            Transform(prod_tex, final_zeta)
        )
        self.wait(2)