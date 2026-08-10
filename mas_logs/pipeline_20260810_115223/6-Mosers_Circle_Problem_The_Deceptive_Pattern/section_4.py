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
        lines = [
            "The true formula is (n choose 4) + (n choose 2) + 1.",
            "Each chord intersection adds exactly one new region.",
            "This accounts for complex overlaps inside the circle.",
            "The simple pattern 2^(n-1) fails us here.",
            "Combinatorics reveals the precise geometric truth."
        ]
        self.setup_layout("Unveiling the True Formula", lines)
        
        # Assets
        circle_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/circle.svg")
        
        # Colors for lecture lines
        colors = [BLUE, GREEN, YELLOW, RED, PURPLE]
        
        # === Animation for Lecture Line 1 ===
        formula = MathTex(r"R = \binom{n}{4} + \binom{n}{2} + 1", font_size=36)
        self.place_in_area(formula, 'B4', 'C6', scale_factor=0.6)
        self.place_at_grid(circle_icon, 'B2', scale_factor=0.5)
        self.play(FadeIn(circle_icon), Write(formula))
        self.lecture[0].set_color(colors[0])
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        term1 = MathTex(r"\binom{n}{2}", color=GREEN, font_size=36)
        self.place_at_grid(term1, 'D3', scale_factor=0.7)
        self.play(FadeIn(term1), formula.animate.set_color(WHITE))
        self.lecture[1].set_color(colors[1])
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        term2 = MathTex(r"\binom{n}{4}", color=YELLOW, font_size=36)
        self.place_at_grid(term2, 'D5', scale_factor=0.7)
        self.play(FadeIn(term2))
        self.lecture[2].set_color(colors[2])
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        fail_text = Text("Pattern 2^(n-1) fails!", color=RED, font_size=24)
        self.place_at_grid(fail_text, 'E3', scale_factor=0.8)
        self.play(Write(fail_text))
        self.lecture[3].set_color(colors[3])
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(colors[4])
        self.play(Indicate(formula))
        self.wait(2)
