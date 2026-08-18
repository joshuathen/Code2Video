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
        self.setup_layout("Synthesis and Application", [
            "Define the target combinatorial polynomial clearly.",
            "Evaluate the polynomial at every root of unity.",
            "Simplify complex power sums to integer results.",
            "Combine values using the inverse filter formula.",
            "Final count emerges from complex arithmetic steps."
        ])
        
        # Assets
        calculator = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        magnifier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifier.svg")

        # === Animation for Lecture Line 1 ===
        poly = MathTex("P(x) = \\sum_{k=0}^{n} \\binom{n}{k}^3 x^k", color="#3498DB")
        self.place_at_grid(poly, 'B2', scale_factor=0.8)
        self.place_at_grid(calculator, 'B5', scale_factor=0.5)
        self.play(Write(poly), FadeIn(calculator))
        self.lecture[0].set_color("#3498DB")

        # === Animation for Lecture Line 2 ===
        eval_points = MathTex("P(\\omega^j) = \\sum \\binom{n}{k}^3 \\omega^{jk}", color="#E74C3C")
        self.place_at_grid(eval_points, 'C2', scale_factor=0.7)
        self.play(FadeIn(eval_points))
        self.lecture[1].set_color("#E74C3C")

        # === Animation for Lecture Line 3 ===
        simplification = MathTex("S = \\frac{1}{3} \\sum_{j=0}^{2} P(\\omega^j)", color="#2ECC71")
        self.place_at_grid(simplification, 'D2', scale_factor=0.7)
        self.play(FadeIn(simplification))
        self.lecture[2].set_color("#2ECC71")

        # === Animation for Lecture Line 4 ===
        formula = MathTex("Count = \\frac{P(1) + P(\\omega) + P(\\omega^2)}{3}", color="#FFFFFF")
        self.place_at_grid(formula, 'E2', scale_factor=0.6)
        self.play(FadeIn(formula))
        self.lecture[3].set_color("#FFFFFF")

        # === Animation for Lecture Line 5 ===
        final_val = MathTex("N = 3^{n-1} + 3 \\binom{n}{3n/3} ...", color="#F1C40F")
        self.place_at_grid(final_val, 'F2', scale_factor=0.8)
        self.place_at_grid(magnifier, 'F5', scale_factor=0.5)
        self.play(Flash(final_val), FadeIn(magnifier))
        self.lecture[4].set_color("#F1C40F")
        self.wait(2)
