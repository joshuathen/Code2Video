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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Roots of Unity Filter Formula", [
            'Evaluate the polynomial at each root of unity.', 
            'Averaging these values isolates the desired terms.', 
            'Unwanted coefficients cancel out across the complex circle.', 
            'We are left with the sum of specific terms.', 
            'This is the elegant Roots of Unity Filter formula.'
        ])
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # 1. Write the polynomial P(x) = a_0 + a_1 x + a_2 x^2 +... in #FFFFFF.
        poly = Text("P(x) = a0 + a1 x + a2 x^2 + ...", font_size=20, color=WHITE)
        self.place_at_grid(poly, 'A5', scale_factor=1.0)
        
        # Setup complex plane group (Axes + Circle + 4 dots)
        axes = VGroup(
            Line(LEFT*1.5, RIGHT*1.5, stroke_width=1, color=GRAY),
            Line(UP*1.5, DOWN*1.5, stroke_width=1, color=GRAY)
        )
        circle = Circle(radius=1.2, color=BLUE_B)
        dots = VGroup(*[Dot(circle.point_at_angle(i * TAU / 4), color=YELLOW) for i in range(4)])
        complex_plane_group = VGroup(axes, circle, dots)
        
        # Issue 36: Anchor 'complex_plane_group' to the grid
        self.place_in_area(complex_plane_group, 'B2', 'E5', scale_factor=0.8)
        
        # Issue 37 & 38: Reposition labels
        label_w0 = Text("w^0", font_size=24)
        label_w1 = Text("w^1", font_size=24)
        label_w2 = Text("w^2", font_size=24)
        label_w3 = Text("w^3", font_size=24)
        
        self.place_at_grid(label_w1, 'A3', scale_factor=0.6) # Issue 37 Fix
        self.place_at_grid(label_w3, 'F3', scale_factor=0.6) # Issue 37 Fix
        self.place_at_grid(label_w2, 'C1', scale_factor=0.6) # Issue 38 Fix
        self.place_at_grid(label_w0, 'C5', scale_factor=0.6)
        
        # Display a column of evaluations P(1), P(omega),..., P(omega^(n-1)) in #00FFFF.
        evals = VGroup(
            Text("P(w^0)", font_size=20, color="#00FFFF"),
            Text("P(w^1)", font_size=20, color="#00FFFF"),
            Text("P(w^2)", font_size=20, color="#00FFFF"),
            Text("P(w^3)", font_size=20, color="#00FFFF")
        ).arrange(DOWN, buff=0.15)
        self.place_at_grid(evals, 'D6', scale_factor=1.0)
        
        self.play(Write(poly))
        self.play(Create(complex_plane_group))
        self.play(Write(label_w0), Write(label_w1), Write(label_w2), Write(label_w3))
        self.play(Write(evals))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        
        # 2. Show the average (1/n) * sum of these evaluations in a #FFFF00 box.
        avg_formula = Text("Sum = 1/4 * [P(w^0) + ... + P(w^3)]", font_size=20, color=YELLOW)
        self.place_at_grid(avg_formula, 'F5', scale_factor=1.0)
        box = SurroundingRectangle(avg_formula, color=YELLOW, buff=0.1)
        
        self.play(Write(avg_formula), Create(box))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(BLUE))
        
        # 3. Expand the sum to show individual coefficients multiplied by sums of powers of omega.
        expansion = Text("= sum a_k * [1/4 sum (w^j)^k]", font_size=20, color=WHITE)
        self.place_at_grid(expansion, 'B5', scale_factor=1.0)
        
        self.play(FadeOut(evals), FadeOut(poly))
        self.play(Write(expansion))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FF00"))
        
        # 4. Highlight terms where index k is a multiple of n in #00FF00, showing others cancel [Asset: ...none.svg].
        term_k_mult = Text("k = 4m: a_k * (1)", font_size=18, color="#00FF00")
        term_k_other = Text("k != 4m: a_k * (0)", font_size=18, color=RED)
        
        self.place_at_grid(term_k_mult, 'A5', scale_factor=1.0)
        self.place_at_grid(term_k_other, 'E1', scale_factor=1.0)
        
        # Asset integration (Issue 27/45)
        cancel_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/none.svg")
        self.place_at_grid(cancel_icon, 'D1', scale_factor=0.3)
        
        self.play(Write(term_k_mult))
        self.play(Write(term_k_other), FadeIn(cancel_icon))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        
        # Display the final Roots of Unity Filter formula.
        final_formula = Text("Sum a_{4m} = 1/4 sum_{j=0}^3 P(w^j)", font_size=22, color=YELLOW)
        final_box = SurroundingRectangle(final_formula, color=YELLOW)
        
        # Final cleanup and formula display
        self.play(
            FadeOut(expansion), FadeOut(term_k_mult), FadeOut(term_k_other), 
            FadeOut(cancel_icon), FadeOut(avg_formula), FadeOut(box)
        )
        self.place_in_area(VGroup(final_formula, final_box), 'B4', 'C6', scale_factor=1.1)
        self.play(Write(final_formula), Create(final_box))
        self.wait(3)
