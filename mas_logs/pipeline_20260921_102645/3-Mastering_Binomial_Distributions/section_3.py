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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Calculate P of X equaling k successes.",
            "Combinations count total ways to arrange successes.",
            "Multiply by probability of successes and failures.",
            "Formula combines these two distinct components.",
            "This defines the full binomial distribution."
        ]
        self.setup_layout("The Binomial Formula Breakdown", lecture_lines)
        
        formula = MathTex(
            "P(X=k) = ", "C(n,k)", " \\cdot ", "p^k \\cdot (1-p)^{n-k}",
            font_size=40
        )
        
        # Asset Loading
        coin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        dice = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dice.svg")

        # === Animation for Lecture Line 1 ===
        # Positioned per constraint (overlap fix 41)
        self.place_in_area(formula, 'D3', 'F6', scale_factor=0.9)
        self.place_at_grid(coin, 'B3', scale_factor=0.5)
        self.play(Write(formula), FadeIn(coin), Indicate(self.lecture[0], color=YELLOW))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.play(formula[1].animate.set_color(TEAL), Indicate(self.lecture[1], color=TEAL))
        self.lecture[1].set_color(TEAL)

        # === Animation for Lecture Line 3 ===
        self.play(formula[3].animate.set_color(ORANGE), Indicate(self.lecture[2], color=ORANGE))
        self.lecture[2].set_color(ORANGE)

        # === Animation for Lecture Line 4 ===
        self.play(Flash(formula), Indicate(self.lecture[3], color=WHITE))
        self.lecture[3].set_color(WHITE)

        # === Animation for Lecture Line 5 ===
        # Color formula white, display dice
        self.place_at_grid(dice, 'E3', scale_factor=0.5)
        self.play(
            formula.animate.set_color(WHITE), 
            FadeIn(dice),
            Indicate(self.lecture[4], color=WHITE)
        )
        self.lecture[4].set_color(WHITE)
        self.wait(2)
