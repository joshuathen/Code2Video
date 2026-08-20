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
            "Bayes' Theorem reverses conditional probability direction.",
            "It calculates cause from observed effects.",
            "We update our belief with new evidence.",
            "The formula uses prior and evidence probabilities.",
            "It turns data into refined knowledge."
        ]
        self.setup_layout("Introduction to Bayes' Theorem", lecture_lines)
        
        # Load Assets
        magnifying_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        book = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/book.svg")
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        formula = MathTex(
            r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}"
        )
        self.place_in_area(formula, 'B3', 'E6', scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FFFF00")
        
        # Add magnifying glass
        self.place_at_grid(magnifying_glass, "B2", scale_factor=0.5)
        
        self.play(
            formula.animate.set_color_by_tex("P(B|A)", "#FFFF00"),
            FadeIn(magnifying_glass)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")
        self.play(formula.animate.set_color_by_tex("P(A)", "#00FFFF"), 
                  formula.animate.set_color_by_tex("P(B)", "#00FFFF"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(GREEN)
        self.play(Indicate(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(PURPLE)
        
        # Add book before fading
        self.place_at_grid(book, "E2", scale_factor=0.5)
        self.add(book)
        
        self.play(
            FadeOut(formula), 
            FadeOut(magnifying_glass),
            FadeOut(book),
            FadeOut(self.lecture),
            FadeOut(self.title)
        )
        self.wait(1)
