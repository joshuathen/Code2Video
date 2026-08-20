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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Connecting Bayes and Independence", [
            "If B is independent of A, B provides nothing.",
            "Bayes' theorem simplifies when independence applies.",
            "P(A|B) becomes P(A) because evidence is uninformative."
        ])
        
        # Formula: P(A|B) = P(B|A) * P(A) / P(B)
        formula = MathTex(
            "P(A|B) = \\frac{P(B|A) \\cdot P(A)}{P(B)}"
        ).set_color(WHITE)
        # Using SVGMobject placeholder per instruction
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        self.place_in_area(formula, 'B2', 'B5', scale_factor=0.9)
        self.add(formula)

        # === Animation for Lecture Line 1 ===
        # If B is independent of A, B provides nothing.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # === Animation for Lecture Line 2 ===
        # Bayes' theorem simplifies when independence applies.
        self.play(self.lecture[1].animate.set_color(ORANGE))
        
        # Simplify to P(A|B) = P(A)
        simplified_formula = MathTex(
            "P(A|B) = P(A)"
        ).set_color(ORANGE)
        self.place_in_area(simplified_formula, 'C2', 'C5', scale_factor=0.8)
        self.play(Transform(formula.copy(), simplified_formula))
        
        # === Animation for Lecture Line 3 ===
        # P(A|B) becomes P(A) because evidence is uninformative.
        self.play(self.lecture[2].animate.set_color("#32CD32"))
        
        # Highlight independence: P(B|A) = P(B)
        independence = MathTex(
            "P(B|A) = P(B)"
        ).set_color("#32CD32")
        self.place_in_area(independence, 'D2', 'D5', scale_factor=0.8)
        self.play(Write(independence))
        
        self.wait(2)
