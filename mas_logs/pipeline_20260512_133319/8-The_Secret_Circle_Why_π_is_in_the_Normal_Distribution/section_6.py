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

class Section6Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Since the squared area is pi, our area is root-pi.",
            "This root-two-pi factor ensures the total probability is one.",
            "Pi is the mathematical footprint of circular symmetry."
        ]
        self.setup_layout("Closing the Loop: Back to 1D", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Formula: I^2 = pi
        # Using Unicode: \u00B2 for squared, \u03C0 for pi
        # Issue 41: Fix grid utilization (B3-C4)
        formula_i2 = Text("I\u00B2 = \u03C0", font_size=48, color=WHITE)
        self.place_in_area(formula_i2, "B3", "C4")
        
        self.play(Write(formula_i2))
        self.wait(1)
        
        # Transition to I = sqrt(pi)
        # Using Unicode: \u221A for square root
        # Issue 41: Fix grid utilization (B3-C4)
        formula_i = Text("I = \u221A\u03C0", font_size=48, color=WHITE)
        self.place_in_area(formula_i, "B3", "C4")
        
        self.play(ReplacementTransform(formula_i2, formula_i))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(GRAY),
            self.lecture[1].animate.set_color(WHITE)
        )
        
        # Normal Distribution Formula
        # Issue 42: Expanded area for pdf_text (B1-C6)
        pdf_text = MarkupText(
            'f(x) = <span color="#FFFFFF">1</span> / <span color="#00FFFF">\u03C3\u221A(2\u03C0)</span> <span color="#FFFFFF">e<sup>-(x-\u03BC)\u00B2 / 2\u03C3\u00B2</sup></span>',
            font_size=24
        )
        self.place_in_area(pdf_text, "B1", "C6")
        
        # Fade out previous formula
        self.play(FadeOut(formula_i))
        self.play(FadeIn(pdf_text))
        
        # Highlight the normalization constant sigma*sqrt(2pi)
        # pdf_text[10:16] corresponds to sigma*sqrt(2pi)
        highlight_box = SurroundingRectangle(pdf_text[10:16], color=YELLOW, buff=0.1)
        self.play(Create(highlight_box))
        self.wait(2)
        self.play(FadeOut(highlight_box), FadeOut(pdf_text))

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(GRAY),
            self.lecture[2].animate.set_color(WHITE)
        )
        
        # Issue 26: Asset Integration
        # 1. Circular Target SVG
        target_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/target.svg")
        target_svg.set_color(BLUE)
        self.place_in_area(target_svg, "B2", "E5", scale_factor=2.0)
        
        # 2. Bell Curve Silhouette overlaying target
        curve = FunctionGraph(
            lambda x: 1.5 * np.exp(-x**2),
            x_range=[-2.5, 2.5],
            color=WHITE
        ).move_to(target_svg.get_center())
        
        # 3. Archer Robin SVG
        # Issue 43: Fix spatial crowding by moving to E1
        robin_svg = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/archer.svg")
        robin_svg.set_color(WHITE)
        self.place_at_grid(robin_svg, "E1", scale_factor=0.8)
        
        # 4. Arrow pointing from circularity to the curve
        arrow = Arrow(
            start=self.grid["D1"], 
            end=target_svg.get_left(), 
            color=YELLOW,
            buff=0.1
        )
        
        self.play(
            FadeIn(target_svg),
            FadeIn(robin_svg),
            run_time=1.5
        )
        self.play(Create(curve), GrowArrow(arrow), run_time=2)
        self.wait(1)
        
        # Final emphasis on the pi symbol
        pi_symbol = Text("\u03C0", font_size=60, color=YELLOW)
        self.place_at_grid(pi_symbol, "C4")
        self.play(Write(pi_symbol), target_svg.animate.set_stroke(opacity=0.3))
        
        self.wait(3)
