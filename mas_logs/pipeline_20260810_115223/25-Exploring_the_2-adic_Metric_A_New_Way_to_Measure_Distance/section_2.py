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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Introducing the 2-adic Metric", [
            "2-adic valuation counts prime factors of 2.",
            "High powers of 2 mean smaller norms.",
            "Compare 8 as tiny and 1/8 as large."
        ])
        
        # Assets
        calc = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        mag = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        scale_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg")
        
        # === Animation for Lecture Line 1 ===
        # 2-adic valuation counts prime factors of 2.
        self.lecture[0].set_color(BLUE)
        
        formula = MathTex("v_2(x) = \\max\\{k : 2^k \\mid x\\}").set_color(WHITE)
        self.place_in_area(formula, 'B3', 'B5', scale_factor=0.8)
        self.place_at_grid(calc, 'B2', scale_factor=0.6)
        self.play(FadeIn(calc), Write(formula))
        
        # === Animation for Lecture Line 2 ===
        # High powers of 2 mean smaller norms.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(GREEN)
        
        norm_def = MathTex("|x|_2 = 2^{-v_2(x)}").set_color(WHITE)
        self.place_at_grid(norm_def, 'C3', scale_factor=0.8)
        self.place_at_grid(ruler, 'C2', scale_factor=0.6)
        self.place_at_grid(mag, 'C4', scale_factor=0.6)
        self.play(FadeIn(ruler), FadeIn(mag), Write(norm_def))
        
        # === Animation for Lecture Line 3 ===
        # Compare 8 as tiny and 1/8 as large.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        val_label = Text("2-adic distance", font_size=24, color=YELLOW)
        self.place_at_grid(val_label, 'D3', scale_factor=0.75)
        self.place_at_grid(scale_icon, 'D2', scale_factor=0.6)
        
        comparison = VGroup(
            MathTex("|8|_2 = 2^{-3} = 1/8"),
            MathTex("|1/8|_2 = 2^{3} = 8")
        ).arrange(DOWN)
        
        self.place_in_area(comparison, 'E2', 'E5', scale_factor=0.7)
        self.play(FadeIn(val_label), FadeIn(scale_icon), Write(comparison))
        self.wait(2)
