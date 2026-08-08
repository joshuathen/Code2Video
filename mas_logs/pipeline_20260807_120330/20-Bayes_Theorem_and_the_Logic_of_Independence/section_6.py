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
        # Colors
        color_posterior = "#FF00FF"  # Magenta
        color_likelihood = "#00FF00" # Green
        color_prior = "#FFFF00"      # Yellow
        color_evidence = "#00FFFF"   # Cyan
        color_highlight = WHITE

        # Setup layout
        self.setup_layout(
            "Independence in Bayes' Context",
            [
                "What if evidence and hypothesis are independent?",
                "The likelihood ratio simplifies to exactly one.",
                "Our posterior belief stays identical to our prior."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_highlight)
        
        # Formula: P(A|B) = [P(B|A) * P(A)] / P(B)
        # Using MathTex with parts for coloring and animation
        # Index breakdown for MathTex("P(A|B)", "=", "{{P(B|A)}}", "\cdot", "{{P(A)}}", "\\over", "{{P(B)}}"):
        # 0: P(A|B)
        # 1: =
        # 2: P(B|A)
        # 3: \cdot
        # 4: P(A)
        # 5: \over (line)
        # 6: P(B)
        formula = MathTex(
            "P(A|B)", "=", "{{P(B|A)}}", "\\cdot", "{{P(A)}}", "\\over", "{{P(B)}}",
            font_size=42
        )
        # Set colors
        formula[0].set_color(color_posterior)
        formula[2].set_color(color_likelihood)
        formula[4].set_color(color_prior)
        formula[6].set_color(color_evidence)
        
        # [Issue 45] Adjusted placement: B3 to D6, scale 1.0
        self.place_in_area(formula, 'B3', 'D6', scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(color_highlight)
        
        # If independent, P(B|A) = P(B)
        new_likelihood = MathTex("P(B)", font_size=42, color=color_evidence)
        new_likelihood.move_to(formula[2].get_center())
        
        # Replacement animation
        self.play(
            formula[2].animate.shift(UP * 0.5).set_opacity(0),
            FadeIn(new_likelihood, shift=UP * 0.5),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(color_highlight)
        
        # Strike through P(B) in numerator and denominator
        # We need to position these relative to the objects
        strike_num = Line(
            new_likelihood.get_left() + LEFT*0.1, 
            new_likelihood.get_right() + RIGHT*0.1, 
            color=RED, stroke_width=4
        )
        strike_den = Line(
            formula[6].get_left() + LEFT*0.1, 
            formula[6].get_right() + RIGHT*0.1, 
            color=RED, stroke_width=4
        )
        
        self.play(Create(strike_num), Create(strike_den))
        self.wait(1)
        
        # Final result: P(A|B) = P(A)
        final_formula = MathTex(
            "P(A|B)", "=", "P(A)",
            font_size=42
        )
        final_formula[0].set_color(color_posterior)
        final_formula[2].set_color(color_prior)
        
        # [Issue 45] Adjusted placement: C3 to C6, scale 1.0
        self.place_in_area(final_formula, 'C3', 'C6', scale_factor=1.0)
        
        self.play(
            FadeOut(VGroup(new_likelihood, strike_num, formula[6], strike_den, formula[3], formula[5])),
            Transform(formula[0], final_formula[0]),
            Transform(formula[1], final_formula[1]),
            Transform(formula[4], final_formula[2])
        )
        self.wait(2)
