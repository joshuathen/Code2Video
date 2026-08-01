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
        # 1. Setup Layout
        title = "The Formal Equation: Mapping Geometry to Algebra"
        lecture_lines = [
            "- Let’s translate this visual stretching into a formal equation.",
            "- Bayes' Theorem calculates the ratio of these specific areas.",
            "- The numerator is the \"True Positive\" region we kept.",
            "- The denominator is the total area of the evidence.",
            "- Algebra confirms what our geometry just demonstrated."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Asset path
        bone_svg_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/bone.svg"
        
        # Colors
        C_FORMULA = "#FFFFFF"
        C_NUMERATOR = "#FFD700"  # Gold
        C_DENOMINATOR = "#87CEEB" # Sky Blue
        C_TP = "#32CD32"         # Lime Green
        C_FP = "#DC143C"         # Crimson
        C_RESULT = "#FFA500"     # Orange

        # === Animation for Lecture Line 1 ===
        # Let’s translate this visual stretching into a formal equation.
        self.lecture[0].set_color(C_FORMULA)
        formula = MathTex(
            "P(H|E)", "=", "{ P(E|H)P(H) ", "\\over", " P(E) }",
            color=C_FORMULA
        )
        # Fix Issue 39: Formula position moved to B2-B4
        self.place_in_area(formula, 'B2', 'B4', scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Bayes' Theorem calculates the ratio of these specific areas.
        self.lecture[1].set_color(C_FORMULA)
        
        # Fix Issue 40: Square visual at D3 given area constraint C2-E4
        square = Square(side_length=3.0, stroke_color=WHITE, fill_opacity=0.0)
        self.place_in_area(square, 'C2', 'E4', scale_factor=0.9)
        s = square.side_length
        ul = square.get_corner(UL)
        
        # Probabilities: P(H)=0.2, P(E|H)=0.9 -> TP=0.18. P(H^c)=0.8, P(E|H^c)=0.1 -> FP=0.08
        # Visualizing the proportions within the square
        tp_rect = Rectangle(width=0.2*s, height=0.9*s, fill_color=C_TP, fill_opacity=0.5, stroke_width=1)
        tp_rect.move_to(ul, aligned_edge=UL)
        
        fp_rect = Rectangle(width=0.8*s, height=0.1*s, fill_color=C_FP, fill_opacity=0.5, stroke_width=1)
        fp_rect.move_to(ul + RIGHT * (0.2*s), aligned_edge=UL)
        
        self.play(Create(square))
        self.play(FadeIn(tp_rect), FadeIn(fp_rect))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The numerator is the "True Positive" region we kept.
        self.lecture[2].set_color(C_NUMERATOR)
        
        # Asset Integration Issue 26
        bone_icon = SVGMobject(bone_svg_path).scale(0.15)
        bone_icon.move_to(tp_rect.get_center())
        
        # Highlight numerator in formula
        self.play(formula[2].animate.set_color(C_NUMERATOR))
        
        arrow_num = Arrow(formula[2].get_bottom(), tp_rect.get_top(), color=C_NUMERATOR, buff=0.1)
        self.play(FadeIn(bone_icon), Create(arrow_num))
        self.play(Indicate(tp_rect))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The denominator is the total area of the evidence.
        self.lecture[3].set_color(C_DENOMINATOR)
        
        # Highlight denominator in formula
        self.play(formula[4].animate.set_color(C_DENOMINATOR))
        
        evidence_group = VGroup(tp_rect, fp_rect)
        evidence_outline = SurroundingRectangle(evidence_group, color=C_DENOMINATOR, buff=0.05)
        
        arrow_den = Arrow(formula[4].get_bottom(), evidence_outline.get_top(), color=C_DENOMINATOR, buff=0.1)
        self.play(Create(evidence_outline), Create(arrow_den))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Algebra confirms what our geometry just demonstrated.
        self.lecture[4].set_color(C_RESULT)
        
        # Plug in numbers
        calc = MathTex(
            "P(H|E) = \\frac{0.18}{0.18 + 0.08} = \\frac{0.18}{0.26}",
            color=C_RESULT, font_size=30
        )
        self.place_at_grid(calc, 'E3', scale_factor=0.8)
        
        # Result text - Fix Issue 41 (Post text positioning at F3 with scale 0.8)
        post_text = Text("Result: ≈ 69%", font_size=28, color=C_RESULT)
        self.place_at_grid(post_text, 'F3', scale_factor=0.8)
        
        self.play(FadeOut(arrow_num), FadeOut(arrow_den))
        self.play(Write(calc))
        self.wait(1)
        self.play(Write(post_text))
        
        # Bone grows representing belief update - Issue 26
        self.play(bone_icon.animate.scale(3).set_color(C_RESULT))
        
        self.wait(2)
