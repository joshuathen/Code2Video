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
        lecture_lines = [
            "The PMF calculates probability for exactly k successes.",
            "Formula: P(X=k) = C(n, k) * p^k * (1-p)^(n-k).",
            "Visual histograms show how n and p change shape.",
            "We calculate specific success probabilities.",
            "Parameters define the entire distribution."
        ]
        self.setup_layout("The Probability Mass Function (PMF)", lecture_lines)
        
        # Elements
        formula = MathTex(
            "P(X=k) = ", "C(n, k)", " \\\\cdot ", "p^k", " \\\\cdot ", "(1-p)^{n-k}"
        ).scale(0.9)
        
        # Color specific parts
        formula[3].set_color("#FF6347")
        formula[5].set_color("#4682B4")

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        c_nk = formula[1].copy()
        self.place_at_grid(c_nk, "B2", 1.5)
        self.play(Write(c_nk))
        self.wait(1)

        # Display the p^k term
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        p_k = formula[3].copy()
        self.place_at_grid(p_k, "C2", 1.5)
        self.play(Write(p_k))
        self.wait(1)

        # Display the (1-p)^(n-k) term
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        q_nk = formula[5].copy()
        self.place_at_grid(q_nk, "D2", 1.5)
        self.play(Write(q_nk))
        self.wait(1)

        # Combine all terms into the final PMF formula.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        self.play(FadeOut(c_nk), FadeOut(p_k), FadeOut(q_nk))
        # Fixed positioning per issue instructions
        self.place_in_area(formula, "D1", "F3", scale_factor=0.9)
        self.play(Write(formula))
        self.wait(1)

        # Flash the complete formula
        self.play(Indicate(formula))
        self.wait(2)
        self.lecture[4].set_color(WHITE)
