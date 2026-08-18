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
        self.setup_layout("The Core of Bayes' Theorem", [
            "Bayes' Theorem reverses conditional probabilities.",
            "We predict the cause from observed evidence.",
            "Start with a prior belief.",
            "Update it using new evidence.",
            "The result is a posterior probability."
        ])
        
        formula = MathTex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}")
        self.place_in_area(formula, 'B2', 'D5', scale_factor=1.2)
        
        # Assets
        detective = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/detective.svg")
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        evidence = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/evidence.svg")
        fingerprint = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/fingerprint.svg")

        # === Animation for Lecture Line 1 ===
        self.play(Write(formula))
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(self.place_at_grid(detective, 'A6', scale_factor=0.5)))
        self.play(formula.animate.shift(LEFT * 0.5))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        self.play(Indicate(formula))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(ORANGE))
        self.play(Circumscribe(formula[0][8:11])) # P(A) - prior

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(TEAL))
        self.play(formula[0][3:8].animate.set_color("#00FFFF"))
        self.play(FadeIn(self.place_at_grid(evidence, 'E6', scale_factor=0.5)))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#39FF14"))
        self.play(formula[0][0:5].animate.set_color("#39FF14"))
        self.play(FadeIn(self.place_at_grid(magnifying_glass, 'B6', scale_factor=0.5)))
        self.play(Flash(formula))
        self.add(self.place_at_grid(fingerprint, 'F1', scale_factor=0.5))
