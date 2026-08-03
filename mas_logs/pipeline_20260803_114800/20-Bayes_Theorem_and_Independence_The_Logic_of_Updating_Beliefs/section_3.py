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
        self.setup_layout("The Bridge: Bayes' Theorem Formula", [
            "Bayes' Theorem updates a prior belief with new evidence.",
            "It scales our initial guess by evidence strength.",
            "The formula yields an updated posterior probability."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show "P(A)" labeled "Prior" in #ADD8E6 on the left.
        self.lecture[0].set_color("#ADD8E6")
        
        prior_prob = MathTex("P(A)", color="#ADD8E6")
        prior_label = Text("Prior", color="#ADD8E6", font_size=24)
        
        self.place_at_grid(prior_prob, "B2", scale_factor=1.2)
        # Fix Issue 25: scale_factor reduced to 0.6 for visual balance
        self.place_at_grid(prior_label, "C2", scale_factor=0.6)
        
        self.play(Write(prior_prob), FadeIn(prior_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # An arrow labeled "Evidence P(B|A) / P(B)" (#FFFFFF) moves from "Prior" to a new term "P(A|B)".
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFFFF")
        
        posterior_prob = MathTex("P(A|B)", color="#FFFFFF")
        self.place_at_grid(posterior_prob, "B5", scale_factor=1.2)
        
        # Grid positions for arrow
        start_point = self.grid["B2"] + RIGHT * 0.5
        end_point = self.grid["B5"] + LEFT * 0.7
        evidence_arrow = Arrow(start=start_point, end=end_point, color="#FFFFFF", buff=0.1)
        
        evidence_label = MathTex(r"\frac{P(B|A)}{P(B)}", color="#FFFFFF")
        self.place_in_area(evidence_label, "A3", "A4", scale_factor=0.8)
        evidence_label.shift(UP * 0.2)
        
        # Fix Issue 23: Repositioned tag to avoid overlap with arrow
        evidence_tag = Text("Evidence", color="#FFFFFF", font_size=18)
        self.place_in_area(evidence_tag, "A3", "A4", scale_factor=0.6)
        evidence_tag.next_to(evidence_label, DOWN, buff=0.1)
        
        self.play(
            GrowArrow(evidence_arrow),
            Write(evidence_label),
            FadeIn(evidence_tag),
            Write(posterior_prob)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Display the full formula "P(A|B) = [P(B|A) \times P(A)] / P(B)" in center, #FFFF00.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFFF00")
        
        full_formula = MathTex(r"P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}", color="#FFFF00")
        # Fix Issue 24: scale_factor adjusted to 1.0
        self.place_in_area(full_formula, "E2", "F5", scale_factor=1.0)
        
        # Issue 19: Integrated asset based.svg
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/based.svg")
        self.place_at_grid(icon, "E1", scale_factor=0.8)
        
        # Transform the existing components into the final formula
        self.play(
            FadeOut(prior_label),
            FadeOut(evidence_tag),
            ReplacementTransform(VGroup(prior_prob, posterior_prob, evidence_label, evidence_arrow).copy(), full_formula),
            FadeIn(icon)
        )
        self.play(Indicate(full_formula))
        self.wait(3)
