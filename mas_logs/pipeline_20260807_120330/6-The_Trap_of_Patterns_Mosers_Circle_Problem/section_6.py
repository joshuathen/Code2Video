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
        self.setup_layout("The Final Formula and Conclusion", [
            "The true formula combines these intersection and chord counts.",
            "It matches the doubling pattern only until five points.",
            "In math, patterns suggest truths, but proofs confirm them."
        ])
        
        # Colors
        GOLD = "#FFD700"
        HIGHLIGHT_COLOR = "#FF4444"
        SOFT_WHITE = "#F0F0F0"
        ACCENT = "#58C4DD"
        
        # === Animation for Lecture Line 1 ===
        # Formula: R = nCr(n, 4) + nCr(n, 2) + 1
        formula = MathTex("R = \\binom{n}{4} + \\binom{n}{2} + 1", color=GOLD)
        formula_box = SurroundingRectangle(formula, color=GOLD, buff=0.2)
        
        # Load asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/gold.svg
        gold_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/gold.svg")
        self.place_at_grid(gold_icon, 'A6', scale_factor=0.3)
        
        formula_group = VGroup(formula, formula_box)
        # Apply Issue 36: scale factor 0.8
        self.place_in_area(formula_group, 'A2', 'B5', scale_factor=0.8)
        
        self.play(self.lecture[0].animate.set_color(GOLD))
        self.play(Create(formula_box), Write(formula), FadeIn(gold_icon))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Display comparison table
        table = MathTable(
            [["1", "1", "1"],
             ["2", "2", "2"],
             ["3", "4", "4"],
             ["4", "8", "8"],
             ["5", "16", "16"],
             ["6", "32", "31"]],
            col_labels=[MathTex("n"), MathTex("2^{n-1}"), MathTex("\\text{Formula}")],
            include_outer_lines=True
        )
        table.get_entries().set_color(WHITE)
        table.get_labels().set_color(ACCENT)
        
        # Apply Issue 34: area 'C1' to 'F5', scale 0.55
        self.place_in_area(table, 'C1', 'F5', scale_factor=0.55)
        
        self.play(self.lecture[1].animate.set_color(ACCENT))
        self.play(Write(table))
        self.wait(1)
        
        # Highlight n=6 divergence
        # table.get_rows()[6] is the last row [6, 32, 31]
        diverge_rect = SurroundingRectangle(table.get_rows()[6], color=HIGHLIGHT_COLOR)
        diverge_label = Text("Diverges!", font_size=24, color=HIGHLIGHT_COLOR, weight=BOLD)
        self.place_at_grid(diverge_label, 'F6', scale_factor=0.8)
        
        self.play(Create(diverge_rect))
        self.play(Write(diverge_label))
        self.wait(3)

        # === Animation for Lecture Line 3 ===
        # Fade out math and display closing quote
        quote_text = "“In mathematics, a pattern is a hint, not a law.”"
        quote = Text(quote_text, font_size=28, color=SOFT_WHITE, slant=ITALIC)
        
        # Apply Issue 35: area 'C1' to 'E6', scale 0.85
        self.place_in_area(quote, 'C1', 'E6', scale_factor=0.85)

        self.play(self.lecture[2].animate.set_color(SOFT_WHITE))
        self.play(
            FadeOut(formula_group),
            FadeOut(gold_icon),
            FadeOut(table),
            FadeOut(diverge_rect),
            FadeOut(diverge_label)
        )
        
        self.play(FadeIn(quote))
        self.wait(5)
