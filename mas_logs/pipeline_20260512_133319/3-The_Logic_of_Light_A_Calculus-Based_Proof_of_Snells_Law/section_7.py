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

class Section7Scene(TeachingScene):
    def construct(self):
        # Setup the scene layout
        lecture_lines = [
            "Light speed in a medium depends on index n.",
            "Replace velocity with the refractive index formula.",
            "The constant speed of light cancels from both sides.",
            "Rearrange the terms into their final form.",
            "We have derived Snell's Law from first principles."
        ]
        self.setup_layout("The Final Arrival: Snell’s Law", lecture_lines)

        # Helper to create fractions without Tex
        def create_fraction(top_text, bottom_text, color=WHITE, font_size=24):
            top = Text(top_text, font_size=font_size, color=color)
            bottom = Text(bottom_text, font_size=font_size, color=color)
            line_width = max(top.width, bottom.width) + 0.1
            line = Line(LEFT, RIGHT, color=color).set_length(line_width)
            return VGroup(top, line, bottom).arrange(DOWN, buff=0.1)

        def create_equation(left_mobj, right_mobj, color=WHITE, font_size=24):
            equal = Text("=", font_size=font_size, color=color)
            return VGroup(left_mobj, equal, right_mobj).arrange(RIGHT, buff=0.3)

        # === Animation for Lecture Line 1 ===
        # Formula: v = c / n
        self.lecture[0].set_color(YELLOW)
        v_label = Text("v", font_size=24)
        c_n_frac = create_fraction("c", "n")
        v_formula = create_equation(v_label, c_n_frac)
        # Issue 47 Fix: Moved lower to avoid title crowding
        self.place_in_area(v_formula, 'B3', 'C4', scale_factor=1.0)
        
        self.play(FadeIn(v_formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Start with derivative equation: sin θ₁ / v₁ = sin θ₂ / v₂
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        lhs_v = create_fraction("sin θ₁", "v₁")
        rhs_v = create_fraction("sin θ₂", "v₂")
        eq_v = create_equation(lhs_v, rhs_v)
        self.place_in_area(eq_v, "C2", "D5", scale_factor=1.1)
        
        self.play(Write(eq_v))
        self.wait(1)

        # Substitution: Replace v1, v2 with c/n1, c/n2
        lhs_sub = create_fraction("sin θ₁", "(c / n₁)")
        rhs_sub = create_fraction("sin θ₂", "(c / n₂)")
        eq_sub = create_equation(lhs_sub, rhs_sub)
        # Issue 48 Fix: Reduced scale for better clarity of nested fractions
        self.place_in_area(eq_sub, 'C2', 'D5', scale_factor=0.9)
        
        self.play(ReplacementTransform(eq_v, eq_sub))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Rearrange to show n1 sin θ1 / c = n2 sin θ2 / c
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        lhs_c = create_fraction("n₁ sin θ₁", "c")
        rhs_c = create_fraction("n₂ sin θ₂", "c")
        eq_c = create_equation(lhs_c, rhs_c)
        self.place_in_area(eq_c, "C2", "D5", scale_factor=1.1)
        
        self.play(ReplacementTransform(eq_sub, eq_c))
        self.wait(0.5)
        
        # Highlight 'c' for cancellation
        c_cancel_l = Line(LEFT, RIGHT, color=RED).set_length(0.3).move_to(lhs_c[2])
        c_cancel_r = Line(LEFT, RIGHT, color=RED).set_length(0.3).move_to(rhs_c[2])
        self.play(Create(c_cancel_l), Create(c_cancel_r))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Final form: n1 sin θ1 = n2 sin θ2
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        final_lhs = Text("n₁ sin θ₁", font_size=28)
        final_rhs = Text("n₂ sin θ₂", font_size=28)
        final_eq = create_equation(final_lhs, final_rhs, font_size=28)
        # Issue 49 Fix: Reduced scale to avoid crowding at bottom
        self.place_in_area(final_eq, 'E2', 'F5', scale_factor=1.1)
        
        self.play(ReplacementTransform(eq_c, final_eq), FadeOut(c_cancel_l), FadeOut(c_cancel_r))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Glow effect
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        final_eq_gold = final_eq.copy().set_color("#FFD700")
        glow = final_eq_gold.copy().set_stroke(width=8, opacity=0.4)
        
        self.play(
            final_eq.animate.set_color("#FFD700"),
            FadeIn(glow)
        )
        self.play(Indicate(final_eq, color="#FFD700", scale_factor=1.1))
        self.wait(2)
