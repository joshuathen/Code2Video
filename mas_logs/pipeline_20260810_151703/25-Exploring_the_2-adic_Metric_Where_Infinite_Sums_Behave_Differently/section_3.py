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
        self.setup_layout("Infinite Sums in the 2-adic World", [
            "A series converges if terms shrink 2-adically.",
            "The ultrametric inequality allows unique behavior.",
            "1+2+4+8... converges to -1."
        ])
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/abacus.svg
        abacus = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/abacus.svg")
        self.place_at_grid(abacus, "A4", scale_factor=0.3)
        self.play(FadeIn(abacus))

        # === Animation for Lecture Line 1 ===
        # A series converges if terms shrink 2-adically.
        self.play(self.lecture[0].animate.set_color("#44AAFF"))
        formula = MathTex(r"|a_n|_2 \to 0").set_color("#44AAFF")
        self.place_at_grid(formula, "B4", scale_factor=0.8)
        self.play(Write(formula))

        # === Animation for Lecture Line 2 ===
        # The ultrametric inequality allows unique behavior.
        self.play(self.lecture[1].animate.set_color("#44AAFF"))
        ineq = MathTex(r"|a+b|_2 \le \max(|a|_2, |b|_2)").set_color("#44AAFF")
        self.place_at_grid(ineq, "C4", scale_factor=0.7)
        self.play(FadeIn(ineq))

        # === Animation for Lecture Line 3 ===
        # 1+2+4+8... converges to -1.
        self.play(self.lecture[2].animate.set_color("#FFAA44"))
        sum_eq = MathTex(r"1 + 2 + 4 + 8 + \dots = -1").set_color("#FFAA44")
        self.place_at_grid(sum_eq, "E4", scale_factor=0.9)
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg
        calc = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        self.place_at_grid(calc, "F4", scale_factor=0.3)
        
        self.play(Flash(sum_eq, color="#FFAA44"))
        self.play(Write(sum_eq), FadeIn(calc))
        
        self.wait(2)
