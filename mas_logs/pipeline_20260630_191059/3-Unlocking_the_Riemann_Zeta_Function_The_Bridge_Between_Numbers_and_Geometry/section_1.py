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

class Section1Scene(TeachingScene):
    def construct(self):
        # Setup the layout with title and lecture lines
        title = "The Mystery of Primes: Meet the Detective"
        lines = [
            "Prime numbers appear scattered and random across the field.",
            "Meet Detective Prime, searching for patterns in their distribution.",
            "The Riemann zeta function reveals their hidden, secret map."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create a field of scattered white dots representing primes
        np.random.seed(42)
        dot_positions = []
        for _ in range(16): # Using 16 dots to match a 4x4 grid later
            rx = np.random.uniform(1.0, 5.0)
            ry = np.random.uniform(-1.8, 1.2)
            dot_positions.append(np.array([rx, ry, 0]))
            
        dots = VGroup(*[Dot(pos, color=WHITE, radius=0.08) for pos in dot_positions])
        
        self.play(FadeIn(dots, lag_ratio=0.1), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Create Detective Prime icon using the provided asset
        # [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/detective.svg]
        detective = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/detective.svg")
        detective.set_color("#87CEEB")
        
        # Initial position off-screen left
        detective.move_to(self.grid["B1"] + LEFT * 2)
        
        self.play(FadeIn(detective))
        
        # Path for detective to scan the field
        scan_points = [self.grid["B2"], self.grid["D3"], self.grid["C5"], self.grid["E4"]]
        for point in scan_points:
            self.play(detective.animate.move_to(point), run_time=1.5, rate_func=smooth)
            self.wait(0.2)

        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Create the gold formula ζ(s)
        # Using Text instead of MathTex to avoid FileNotFoundError: 'latex' when LaTeX is not installed
        zeta_formula = Text("ζ(s)", color="#FFD700")
        # Fix for Issue 29: Position zeta_formula in area A2-A5
        self.place_in_area(zeta_formula, 'A2', 'A5', scale_factor=1.1)
        
        # Prepare the grid alignment (secret map)
        prime_grid = dots.copy()
        prime_grid.arrange_in_grid(rows=4, cols=4, buff=0.5)
        # Fix for Issue 30: Position the prime grid in area B2-E5
        self.place_in_area(prime_grid, 'B2', 'E5', scale_factor=0.9)
        
        # Final detective position
        # Fix for Issue 31: Move detective to F3
        detective_final = detective.copy()
        self.place_at_grid(detective_final, 'F3', scale_factor=0.6)

        # Animate transition
        self.play(
            FadeIn(zeta_formula, shift=UP),
            Transform(dots, prime_grid),
            Transform(detective, detective_final),
            run_time=3
        )
        
        self.wait(3)
        
        # Final cleanup for the section
        self.lecture[2].set_color(WHITE)
        self.wait(1)
