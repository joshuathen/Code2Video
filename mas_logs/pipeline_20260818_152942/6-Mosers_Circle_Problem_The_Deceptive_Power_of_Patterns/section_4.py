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
        lecture_lines = [
            "Euler found the actual formula.",
            "Regions equal n choose 4.",
            "Plus n choose 2, plus 1.",
            "This defines every internal intersection.",
            "Structure proves truth beyond sequences."
        ]
        self.setup_layout("The General Formula: Euler’s Insight", lecture_lines)
        
        formula = MathTex(r"R_n = \binom{n}{4} + \binom{n}{2} + 1", font_size=42)
        self.place_in_area(formula, 'B3', 'D4', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(FadeIn(formula))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        # Highlighting n choose 4 part
        highlight1 = SurroundingRectangle(formula[0][4:9], color="#FFD700", buff=0.1)
        self.place_in_area(highlight1, 'B3', 'D4', scale_factor=0.95)
        self.play(Create(highlight1))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00CED1"))
        highlight2 = SurroundingRectangle(formula[0][10:15], color="#00CED1", buff=0.1)
        self.place_in_area(highlight2, 'B3', 'D4', scale_factor=0.95)
        self.play(ReplacementTransform(highlight1, highlight2))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFFFFF"))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF4500"))
        self.play(FadeOut(highlight2))
        self.play(formula.animate.scale(1.1).set_color("#FF4500"))
        
        formula_label = Text("General formula for Regions", font_size=24)
        self.place_at_grid(formula_label, 'E3', scale_factor=0.7)
        self.play(Write(formula_label))
        self.wait(2)
