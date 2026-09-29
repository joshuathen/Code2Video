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
            "The Zeta function sums 1 over n to the s.",
            "For s greater than 1, it converges beautifully.",
            "Imagine a squirrel gathering acorns forever."
        ]
        self.setup_layout("The Infinite Sum: Intuitive Foundation", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # The Zeta function sums 1 over n to the s.
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        zeta_formula = MathTex(r"\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}")
        self.place_in_area(zeta_formula, 'A2', 'B5', scale_factor=1.0)
        self.play(Write(zeta_formula))

        # === Animation for Lecture Line 2 ===
        # For s greater than 1, it converges beautifully.
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Bars representing 1/n^s convergence + acorns
        acorns = VGroup(*[
            SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/acorn.svg").scale(0.5)
            for _ in range(5)
        ]).arrange(RIGHT, buff=0.2)
        
        self.place_in_area(acorns, 'C2', 'D4', scale_factor=0.9)
        self.play(Create(acorns))
        self.play(acorns.animate.set_color("#FFFF00")) # Glow effect

        # === Animation for Lecture Line 3 ===
        # Imagine a squirrel gathering acorns forever.
        self.play(self.lecture[2].animate.set_color("#FFA500"))
        squirrel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/squirrel.svg")
        self.place_at_grid(squirrel, 'E4', scale_factor=0.9)
        self.play(FadeIn(squirrel))
        self.wait(2)
