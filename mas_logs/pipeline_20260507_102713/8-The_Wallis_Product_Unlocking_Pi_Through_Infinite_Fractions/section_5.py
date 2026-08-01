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

class Section5Scene(TeachingScene):
    def construct(self):
        # Title and Lecture Lines
        title_str = "The Grand Reveal: Wallis's Formula"
        lecture_lines = [
            'Combining the even and odd sequences reveals a ratio.',
            'As n grows, this ratio equals one exactly.',
            'Rearranging the terms isolates Pi halves on one side.',
            "The result is Wallis's famous infinite string of fractions.",
            'Geometry and algebra unite in this simple, infinite beauty.'
        ]
        
        self.setup_layout(title_str, lecture_lines)
        
        # Colors
        COLOR_RATIO = WHITE
        COLOR_PAIRS = "#ADD8E6" # light blue
        COLOR_APPROX = "#FFFF00" # yellow
        COLOR_FINAL = WHITE
        
        # === Animation for Lecture Line 1 ===
        # The expression for the ratio I_{2n}/I_{2n+1} is built
        self.lecture[0].set_color(YELLOW)
        
        # Use simple Text instead of MathTex to avoid LaTeX issues
        ratio_expr = Text("Ratio: I(2n) / I(2n+1)", font_size=24, color=COLOR_RATIO)
        self.place_at_grid(ratio_expr, "A3", scale_factor=1.0)
        
        circle = Circle(radius=0.5, color=BLUE, fill_opacity=0.2)
        square = Square(side_length=1.0, color=RED, fill_opacity=0.1)
        
        # Fix for Issue 32 & 33: Set scale factor to 1.4 as requested
        self.place_at_grid(circle, "B3", scale_factor=1.4)
        self.place_at_grid(square, "B3", scale_factor=1.4)
        
        self.play(FadeIn(ratio_expr), Create(circle), Create(square))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # As n grows, this ratio equals one exactly.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        limit_text = Text("As n \u2192 \u221E , Ratio \u2192 1", font_size=24, color=COLOR_RATIO)
        self.place_at_grid(limit_text, "B5", scale_factor=1.0)
        
        self.play(Write(limit_text))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # Rearranging the terms isolates Pi halves on one side.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        pi_half_expr = Text("\u03C0/2 = Rearranged Product", font_size=24, color=COLOR_RATIO)
        self.place_at_grid(pi_half_expr, "C3", scale_factor=1.0)
        
        self.play(Transform(ratio_expr, pi_half_expr))
        self.wait(1)
        
        # === Animation for Lecture Line 4 ===
        # The result is Wallis's famous infinite string of fractions.
        # The fractions on the right regroup into pairs (2/1 * 2/3) * (4/3 * 4/5) ... 
        # Symbol approx 1.5707 ... appears in yellow
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        pairs_text = Text("(2/1 \u00B7 2/3) \u00B7 (4/3 \u00B7 4/5) \u00B7 (6/5 \u00B7 6/7) ...", font_size=22, color=COLOR_PAIRS)
        self.place_at_grid(pairs_text, "D3", scale_factor=1.0)
        
        approx_val = Text("\u2248 1.5707...", font_size=22, color=COLOR_APPROX)
        self.place_at_grid(approx_val, "D5", scale_factor=1.0)
        
        self.play(Write(pairs_text))
        self.play(FadeIn(approx_val))
        self.wait(1)
        
        # === Animation for Lecture Line 5 ===
        # Final formula pi/2 = prod (4n^2)/(4n^2-1) in glowing white box.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Use Unicode for product and squares
        final_formula_text = Text("\u03C0/2 = \u220F [ 4n\u00B2 / (4n\u00B2 - 1) ]", font_size=28, color=COLOR_FINAL)
        
        # Container box
        box = SurroundingRectangle(final_formula_text, color=WHITE, buff=0.2)
        final_group = VGroup(box, final_formula_text)
        
        # Fix for Issue 31: Use place_in_area for the final formula
        self.place_in_area(final_group, "E1", "F6", scale_factor=0.65)
        
        self.play(Create(box), Write(final_formula_text))
        self.play(Indicate(final_group, color=WHITE))
        
        self.wait(3)
