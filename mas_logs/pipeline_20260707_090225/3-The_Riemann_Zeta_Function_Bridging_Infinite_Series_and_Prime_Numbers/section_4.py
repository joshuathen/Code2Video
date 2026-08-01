from manim import *

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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup the layout using the base class with Stage-3 prompt script
        self.setup_layout(
            "The Golden Key: The Euler Product Formula", 
            [
                'Euler discovered a secret link to prime numbers.', 
                'The sum over all integers equals a product of primes.', 
                'Each prime acts as a filter in this machine.', 
                'This formula encodes the DNA of the prime numbers.', 
                'Zeta is the key to unlocking prime distribution secrets.'
            ]
        )

        # Colors as per instructions
        COLOR_SUM = "#FFFFFF"
        COLOR_PROD = "#00FFFF"
        COLOR_SHUTTER = "#FFFF00"
        COLOR_DNA = "#FFD700"

        # === Animation for Lecture Line 1 ===
        # Euler discovered a secret link to prime numbers.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Zeta sum formula - Issue 48 Fix: Horizontal utilization
        zeta_sum = Text("Σ 1/nˢ", color=COLOR_SUM)
        self.place_in_area(zeta_sum, 'A2', 'B5', scale_factor=1.0)
        
        self.play(Write(zeta_sum))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The sum over all integers equals a product of primes.
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # Euler product formula - Issue 46 Fix: Prevent overlap
        euler_prod = Text("Π (1 - p⁻ˢ)⁻¹", color=COLOR_PROD)
        self.place_in_area(euler_prod, 'C2', 'D5', scale_factor=0.9)

        self.play(FadeIn(euler_prod, shift=DOWN))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Each prime acts as a filter in this machine.
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Transition: Clear the center to visualize the Prime Sieve machine
        # We will keep formulas in the corners temporarily
        self.play(
            zeta_sum.animate.scale(0.5).move_to(self.grid["A6"]), 
            euler_prod.animate.scale(0.5).move_to(self.grid["F6"])
        )
        
        # Static 'shutters' labeled '2', '3', '5' in yellow
        s2 = Text("2", color=COLOR_SHUTTER)
        s3 = Text("3", color=COLOR_SHUTTER)
        s5 = Text("5", color=COLOR_SHUTTER)
        
        self.place_at_grid(s2, "D2", scale_factor=0.8)
        self.place_at_grid(s3, "D3", scale_factor=0.8)
        self.place_at_grid(s5, "D4", scale_factor=0.8)
        
        shutter_lines = VGroup(
            Line(self.grid["D2"]+LEFT*0.3, self.grid["D2"]+RIGHT*0.3, color=COLOR_SHUTTER),
            Line(self.grid["D3"]+LEFT*0.3, self.grid["D3"]+RIGHT*0.3, color=COLOR_SHUTTER),
            Line(self.grid["D4"]+LEFT*0.3, self.grid["D4"]+RIGHT*0.3, color=COLOR_SHUTTER)
        ).shift(DOWN*0.3)
        
        self.play(FadeIn(s2), FadeIn(s3), FadeIn(s5), Create(shutter_lines))

        # Falling integers: 2, 3, 4, 5, 6
        integers = ["2", "3", "4", "5", "6"]
        for i, val in enumerate(integers):
            m = Text(val, font_size=24)
            self.place_at_grid(m, "A3") # Top entrance
            self.play(FadeIn(m), run_time=0.1)
            
            target_pos = self.grid["F3"]
            
            if val == "6":
                # Integer '6' passes shutters '2' and '3' and they glow white
                self.play(m.animate.move_to(self.grid["D3"]), run_time=0.3)
                self.play(
                    m.animate.set_color(WHITE),
                    s2.animate.set_color(WHITE),
                    s3.animate.set_color(WHITE),
                    run_time=0.2
                )
                self.play(
                    s2.animate.set_color(COLOR_SHUTTER),
                    s3.animate.set_color(COLOR_SHUTTER),
                    m.animate.move_to(target_pos),
                    run_time=0.3
                )
            else:
                # Regular fall
                self.play(m.animate.move_to(target_pos), run_time=0.4)
            
            self.play(FadeOut(m), run_time=0.1)

        self.wait(1)
        self.play(FadeOut(s2), FadeOut(s3), FadeOut(s5), FadeOut(shutter_lines), FadeOut(zeta_sum), FadeOut(euler_prod))

        # === Animation for Lecture Line 4 ===
        # This formula encodes the DNA of the prime numbers.
        self.play(self.lecture[3].animate.set_color(YELLOW))
        
        # Display text 'DNA of the Primes' in bold white above merged formula
        dna_label = Text("DNA of the Primes", weight=BOLD, color=WHITE)
        self.place_in_area(dna_label, "B2", "B5", scale_factor=0.9)
        
        # Merge formulas into one gold equation
        dna_formula = Text("ζ(s) = Σ 1/nˢ = Π (1 - p⁻ˢ)⁻¹", color=COLOR_DNA)
        self.place_in_area(dna_formula, "C2", "D5", scale_factor=0.7)
        
        self.play(Write(dna_formula))
        self.play(FadeIn(dna_label, shift=UP))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Zeta is the key to unlocking prime distribution secrets.
        self.play(self.lecture[4].animate.set_color(YELLOW))
        
        # prime_example - Issue 47 Fix: Horizontal area for long text
        prime_example = Text("(1 - 2⁻ˢ)⁻¹ · (1 - 3⁻ˢ)⁻¹ · (1 - 5⁻ˢ)⁻¹ ...", font_size=20)
        self.place_in_area(prime_example, 'E1', 'F6', scale_factor=0.8)
        
        self.play(FadeIn(prime_example, shift=UP))
        
        # Highlighting the golden key
        rect = SurroundingRectangle(dna_formula, color=COLOR_DNA, buff=0.2)
        self.play(Create(rect))
        self.wait(2)
