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
            "Bayes' theorem bridges prior beliefs and new evidence.",
            "Formula: P(A|B) is P(B|A) times P(A) over P(B).",
            "We reverse perspective to find a hidden cause.",
            "It helps us calculate probabilities of hidden states.",
            "Evidence updates our understanding of the world."
        ]
        self.setup_layout("Introduction to Bayes' Theorem", lecture_lines)
        
        # Load SVG Assets
        detective = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/detective.svg")
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")

        # === Animation for Lecture Line 1 ===
        # Fade in P(A|B) and P(B|A) terms
        term1 = MathTex("P(A|B)", color=WHITE)
        term2 = MathTex("P(B|A)", color=WHITE)
        self.place_at_grid(term1, "A2", scale_factor=1.0)
        self.place_at_grid(term2, "A5", scale_factor=1.0)
        self.place_at_grid(detective, "B3", scale_factor=0.6)
        self.play(FadeIn(term1), FadeIn(term2), FadeIn(detective))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Slide P(B|A) * P(A) / P(B) into view
        formula = MathTex("P(A|B) = \\frac{P(B|A) \\cdot P(A)}{P(B)}", color="#FFD700")
        self.place_in_area(formula, "C2", "C5", scale_factor=0.9)
        self.play(FadeIn(formula))
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        # Connect P(A|B) to right side with arrow
        arrow = Arrow(start=term1.get_bottom(), end=formula.get_top(), color="#FFFFFF")
        self.play(Create(arrow))
        self.lecture[2].set_color("#FFFFFF")

        # === Animation for Lecture Line 4 ===
        # Emphasize P(A|B) P(B) = P(B|A) P(A)
        eq2 = MathTex("P(A|B) \\cdot P(B) = P(B|A) \\cdot P(A)", color="#FF6347")
        self.place_in_area(eq2, "D2", "D5", scale_factor=0.8)
        self.play(Write(eq2))
        self.lecture[3].set_color("#FF6347")

        # === Animation for Lecture Line 5 ===
        # Highlight final theorem structure
        highlight = SurroundingRectangle(formula, color="#FFD700", buff=0.1)
        self.place_at_grid(compass, "F5", scale_factor=0.6)
        self.play(Create(highlight), FadeIn(compass))
        self.lecture[4].set_color("#FFD700")
        self.wait(2)
