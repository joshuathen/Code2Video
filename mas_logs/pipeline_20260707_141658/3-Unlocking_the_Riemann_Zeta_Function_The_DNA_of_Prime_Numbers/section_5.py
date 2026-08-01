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

class Section5Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title_text = "The Golden Key: The Euler Product"
        lecture_lines = [
            "Leonhard Euler discovered a bridge to the primes.",
            "The Zeta function equals an infinite product of primes.",
            "It acts like a filter, extracting the prime essence.",
            "This formula links simple counting to the prime numbers.",
            "The Zeta function \"knows\" every prime number in existence."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Leonhard Euler discovered a bridge to the primes.
        # Formula ζ(s) and Product symbol Π appear in white (#FFFFFF).
        self.lecture[0].set_color(WHITE) 
        
        zeta_sym = Text("ζ(s)", color=WHITE, font_size=40)
        prod_sym = Text("Π", color=WHITE, font_size=40)
        self.place_at_grid(zeta_sym, "B3", scale_factor=1.0)
        self.place_at_grid(prod_sym, "B4", scale_factor=1.0)
        
        self.play(FadeIn(zeta_sym), FadeIn(prod_sym))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The Zeta function equals an infinite product of primes.
        self.lecture[1].set_color(WHITE)
        
        # Define formulas and use Issue fixes for consistent positioning
        # Fallback to Text with Unicode due to potential LaTeX environment constraints
        zeta_formula = Text("ζ(s) = Σ 1/nˢ", color=WHITE, font_size=32)
        euler_product = Text("= Π 1/(1 - p⁻ˢ)", color=WHITE, font_size=32)
        
        # Fixing Issue 34, 35, 36: Position using place_in_area and scale 1.0
        self.place_in_area(zeta_formula, 'B2', 'B5', scale_factor=1.0)
        self.place_in_area(euler_product, 'D2', 'D5', scale_factor=1.0)

        self.play(
            ReplacementTransform(zeta_sym, zeta_formula),
            ReplacementTransform(prod_sym, euler_product)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # It acts like a filter, extracting the prime essence.
        # Integer stream (1, 2, 3, 4, 5...) falls through a silver sieve (#C0C0C0).
        self.lecture[2].set_color("#C0C0C0") # Matching silver sieve color
        
        # Move formulas up temporarily to make room for the filter animation
        self.play(
            zeta_formula.animate.scale(0.7).to_edge(UP, buff=1.2),
            euler_product.animate.scale(0.7).next_to(zeta_formula, DOWN, buff=0.1)
        )
        
        sieve = Line(
            start=self.grid["D1"] + LEFT*0.5, 
            end=self.grid["D6"] + RIGHT*0.5, 
            color="#C0C0C0", 
            stroke_width=6
        )
        self.play(Create(sieve))
        
        integers_list = [1, 2, 3, 4, 5, 6]
        int_mobs = VGroup(*[Text(str(i), font_size=30, color=WHITE) for i in integers_list])
        for i, mob in enumerate(int_mobs):
            self.place_at_grid(mob, f"B{i+1}")
            
        self.play(FadeIn(int_mobs, shift=DOWN))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This formula links simple counting to the prime numbers.
        # Sieve filters non-primes; primes (2, 3, 5...) pass and turn green (#00FF00).
        self.lecture[3].set_color("#00FF00") # Matching prime color
        
        primes = [2, 3, 5]
        animations = []
        for i, val in enumerate(integers_list):
            mob = int_mobs[i]
            if val in primes:
                # Primes pass through
                animations.append(mob.animate.move_to(self.grid[f"E{i+1}"]).set_color("#00FF00"))
            else:
                # Non-primes are filtered (fading at sieve line)
                target_pos = sieve.get_center() + np.array([(i-2.5)*0.6, 0, 0])
                animations.append(mob.animate.move_to(target_pos).set_opacity(0))
        
        self.play(*animations, run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The Zeta function "knows" every prime number in existence.
        self.lecture[4].set_color(GOLD) 
        
        # Prime terms like (1-2^-s)^-1 appear and align in a row.
        prime_terms = VGroup(
            Text("(1-2⁻ˢ)⁻¹", font_size=20, color="#00FF00"),
            Text("(1-3⁻ˢ)⁻¹", font_size=20, color="#00FF00"),
            Text("(1-5⁻ˢ)⁻¹", font_size=20, color="#00FF00")
        )
        for i, term in enumerate(prime_terms):
            self.place_at_grid(term, f"F{i+2}")
            
        self.play(FadeIn(prime_terms))
        self.wait(1)
        
        # Terms compress into the final Euler Product formula.
        self.play(
            FadeOut(int_mobs),
            FadeOut(sieve),
            FadeOut(prime_terms)
        )
        
        # Reset formula positions and scales to satisfy the mandatory grid constraints (Issues 34, 35, 36)
        # We call place_in_area again to ensure they end up exactly where the critics requested.
        self.place_in_area(zeta_formula, 'B2', 'B5', scale_factor=1.0)
        self.place_in_area(euler_product, 'D2', 'D5', scale_factor=1.0)
        
        self.play(FadeIn(zeta_formula), FadeIn(euler_product))
        self.play(Indicate(euler_product, color=GOLD))
        self.wait(3)
