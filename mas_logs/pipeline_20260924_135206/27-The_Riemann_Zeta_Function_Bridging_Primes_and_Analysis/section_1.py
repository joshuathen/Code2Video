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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The infinite geometric series sums to one over one-minus-r.",
            "Euler's formula links products over primes to sums.",
            "Imagine a sieve filtering composite numbers away."
        ]
        self.setup_layout("Prerequisite: The Infinite Geometric Series", lecture_lines)
        
        # Elements
        series = MathTex("1", "+", "r", "+", "r^2", "+", "\\dots", "=", "\\frac{1}{1-r}", color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(series, 'A2', 'B5', scale_factor=1.0)
        self.play(FadeIn(series))
        self.play(self.lecture[0].animate.set_color("#ADD8E6"))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        
        # Highlight formula specifically
        formula_part = series[-1]
        rect = SurroundingRectangle(formula_part, color=YELLOW, buff=0.1)
        self.play(Create(rect))
        self.wait(1)
        self.play(FadeOut(rect))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(self.lecture[2].animate.set_color("#90EE90"))
        
        # Create a visual representation for a sieve
        sieve_label = Text("Number Sieve", font_size=24)
        self.place_at_grid(sieve_label, 'D3', scale_factor=0.9)
        self.play(FadeIn(sieve_label))
        
        # Represent numbers 1..10
        numbers = VGroup(*[Text(str(i), font_size=20) for i in range(1, 11)])
        numbers.arrange(RIGHT, buff=0.2)
        self.place_at_grid(numbers, 'E3', scale_factor=0.7)
        self.play(FadeIn(numbers))
        
        # \"Filtering\" composite numbers
        composites = [numbers[3], numbers[5], numbers[7], numbers[8], numbers[9]]
        self.play(*[FadeOut(n) for n in composites])
        self.wait(2)
