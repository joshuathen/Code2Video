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
        # Define Title and Lecture Lines from Storyboard
        title_text = "The Core: Deriving Bayes' Theorem"
        lecture_lines = [
            "Bayes' Theorem allows us to flip conditional probabilities.",
            "We start by relating intersections to conditional probabilities.",
            "By setting these equal, we find a new relationship.",
            "This formula helps us update beliefs with new evidence.",
            "The result is the famous Bayes' Theorem formula."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Color constants
        WHITE_C = "#FFFFFF"
        YELLOW_C = "#FFFF00"
        GREEN_C = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # Display the equation P(A and B) = P(A|B) * P(B) in the top row (A1-A6).
        self.lecture[0].set_color(WHITE_C)
        eq1 = MathTex("P(A \\cap B)", "=", "P(A|B)P(B)", color=WHITE_C)
        self.place_in_area(eq1, "A1", "A6", scale_factor=0.9)
        self.play(Write(eq1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Below it, display the symmetric equation P(A and B) = P(B|A) * P(A) (B1-B6).
        self.lecture[1].set_color(WHITE_C)
        eq2 = MathTex("P(A \\cap B)", "=", "P(B|A)P(A)", color=WHITE_C)
        self.place_in_area(eq2, "B1", "B6", scale_factor=0.9)
        self.play(Write(eq2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Animate an equals sign between the right sides, highlighting both terms in yellow (#FFFF00) at C1-C6.
        self.lecture[2].set_color(YELLOW_C)
        
        combined_eq_mid = MathTex("P(A|B)P(B)", "=", "P(B|A)P(A)", color=YELLOW_C)
        self.place_in_area(combined_eq_mid, "C1", "C6", scale_factor=0.9)

        self.play(
            eq1[2].animate.set_color(YELLOW_C),
            eq2[2].animate.set_color(YELLOW_C),
            ReplacementTransform(eq1[2].copy(), combined_eq_mid[0]),
            ReplacementTransform(eq2[2].copy(), combined_eq_mid[2]),
            Write(combined_eq_mid[1]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Fade out the shared P(A and B) parts and move terms to a single equality at D1-D6.
        self.lecture[3].set_color(WHITE_C)
        
        final_combined_eq = MathTex("P(A|B)P(B)", "=", "P(B|A)P(A)", color=YELLOW_C)
        self.place_in_area(final_combined_eq, "D1", "D6", scale_factor=0.9)
        
        self.play(
            FadeOut(eq1),
            FadeOut(eq2),
            ReplacementTransform(combined_eq_mid, final_combined_eq)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Rearrange terms to show P(A|B) = [P(B|A)P(A)] / P(B) at E1-F6.
        # Include evidence icon [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/evidence.svg].
        self.lecture[4].set_color(GREEN_C)
        
        bayes_formula = MathTex("P(A|B)", "=", "\\frac{P(B|A)P(A)}{P(B)}", color=GREEN_C)
        self.place_in_area(bayes_formula, "E1", "F6", scale_factor=1.1)
        
        # Load and place evidence icon next to the denominator P(B)
        evidence_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/evidence.svg")
        evidence_icon.set_color(GREEN_C)
        evidence_icon.scale(0.3).next_to(bayes_formula[2], RIGHT, buff=0.3)
        
        # Highlight the whole result with a box
        box = SurroundingRectangle(VGroup(bayes_formula, evidence_icon), color=GREEN_C, buff=0.2)
        
        self.play(
            ReplacementTransform(final_combined_eq, bayes_formula),
            FadeIn(evidence_icon)
        )
        self.play(Create(box))
        self.wait(3)
