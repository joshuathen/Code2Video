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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Final Formula", [
            "Plug V and E into Euler's formula.",
            "R equals E minus V plus 1.",
            "R simplifies to nC4 plus nC2 plus 1.",
            "For n=6, the formula gives 31 regions.",
            "The doubling pattern was just a coincidence!"
        ])
        
        # Colors for highlights
        color_line1 = WHITE
        color_line2 = WHITE
        color_line3 = YELLOW
        color_line4 = RED
        color_line5 = RED

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_line1)
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/formula.svg]
        formula_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/formula.svg")
        self.place_at_grid(formula_icon, "A1", scale_factor=0.6)
        
        euler_formula = MathTex("R = E - V + 1", color=WHITE)
        self.place_in_area(euler_formula, "A2", "A6", scale_factor=0.9)
        
        self.play(FadeIn(formula_icon), FadeIn(euler_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color_line2)
        # R equals E minus V plus 1.
        self.play(Indicate(euler_formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(color_line3)
        # R simplifies to nC4 + nC2 + 1
        comb_formula = MathTex("R = \\binom{n}{4} + \\binom{n}{2} + 1", color=YELLOW)
        self.place_in_area(comb_formula, "B1", "B6", scale_factor=0.9)
        self.play(Transform(euler_formula.copy(), comb_formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(color_line4)
        
        # Compute n=5 result as 16 in green (#00FF00).
        n5_calc = MathTex("n=5: \\binom{5}{4} + \\binom{5}{2} + 1 = 5 + 10 + 1 = 16", color=GREEN)
        self.place_in_area(n5_calc, "C1", "C6", scale_factor=0.7)
        self.play(Write(n5_calc))
        self.wait(1)
        
        # Compute n=6 result as 31 in red (#FF0000).
        n6_calc = MathTex("n=6: \\binom{6}{4} + \\binom{6}{2} + 1 = 15 + 15 + 1 = 31", color=RED)
        self.place_in_area(n6_calc, "D1", "D6", scale_factor=0.7)
        self.play(Write(n6_calc))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(color_line5)
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/pattern.svg]
        pattern_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pattern.svg")
        self.place_at_grid(pattern_icon, "E1", scale_factor=0.6)

        failure_note = MathTex("2^{6-1} = 32 \\neq 31", color=RED)
        self.place_in_area(failure_note, "E2", "E6", scale_factor=0.8)
        
        cross_out = Cross(failure_note, stroke_width=4, color=RED)
        
        self.play(FadeIn(pattern_icon), FadeIn(failure_note))
        self.play(Create(cross_out))
        self.wait(2)
