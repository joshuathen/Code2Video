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
        lecture_lines_text = [
            "Bayes' Theorem helps update our prior beliefs.",
            "We use P(A|B) to correct our understanding.",
            "Prior P(A) is updated by evidence P(B).",
            "The evidence tips our internal probability scale.",
            "We reverse prediction using the correction formula."
        ]
        self.setup_layout("Bayes' Theorem: The Logic of Reverse Prediction", lecture_lines_text)
        
        # Setup Mobjects
        formula = MathTex(r"P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}", color=WHITE)
        p_b_a = MathTex(r"P(B|A)", color="#FF9900")
        
        box1 = Square(side_length=1.5, color=WHITE)
        box2 = Square(side_length=1.5, color=WHITE).flip(RIGHT)
        boxes = VGroup(box1, box2).arrange(RIGHT, buff=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FFCC"))
        self.place_in_area(formula, 'B2', 'C5', scale_factor=1.2)
        self.play(Write(formula))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FF9900"))
        self.play(FadeIn(p_b_a))
        self.place_at_grid(p_b_a, 'D3', scale_factor=1.5)
        self.play(p_b_a.animate.scale(1.5))
        self.play(p_b_a.animate.scale(0.66))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#00FFCC"))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color("#FF9900"))
        self.place_in_area(boxes, 'E2', 'F5', scale_factor=0.5)
        self.play(Create(boxes))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color("#00FFCC"))
