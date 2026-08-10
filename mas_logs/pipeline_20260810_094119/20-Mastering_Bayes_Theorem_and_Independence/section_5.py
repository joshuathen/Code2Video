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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Let's apply this to a medical test.",
            "A rare virus requires calculating posterior probability.",
            "The low base rate can cause false positives.",
            "Bayes' theorem reveals the truth behind test results.",
            "Probability updates our understanding of the patient's state."
        ]
        self.setup_layout("Practical Application: The Diagnostic Test", lecture_lines)

        # Create elements
        patient_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/patient.svg")
        virus_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/virus.svg")
        
        tree = VGroup(
            patient_icon,
            Line(ORIGIN, RIGHT*1.5),
            Dot(RIGHT*1.5),
            Line(RIGHT*1.5, RIGHT*3 + UP),
            Line(RIGHT*1.5, RIGHT*3 + DOWN),
            Text("Disease", font_size=18).next_to(RIGHT*3 + UP, UP, buff=0.1),
            Text("No Disease", font_size=18).next_to(RIGHT*3 + DOWN, DOWN, buff=0.1)
        )
        
        formula = VGroup(
            MathTex(r"P(D|+) = \frac{P(+|D)P(D)}{P(+)}", font_size=32),
            virus_icon
        ).arrange(RIGHT, buff=0.2)
        
        result = Text("P(Disease|Positive) is low!", color="#00FF00", font_size=24)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.place_in_area(tree, "A1", "C3", scale_factor=0.6)
        self.play(Create(tree))

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.place_at_grid(formula, "D3", scale_factor=0.8)
        self.play(Write(formula))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.place_at_grid(result, "E4", scale_factor=0.8)
        self.play(FadeIn(result))
        self.wait(2)
