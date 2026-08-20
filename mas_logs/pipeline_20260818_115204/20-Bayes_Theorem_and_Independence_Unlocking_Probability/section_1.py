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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Conditional probability shrinks the sample space.",
            "Formula: P(A|B) equals P(A ∩ B) over P(B).",
            "Visualizing the reduction clarifies the relationship."
        ]
        self.setup_layout("Prerequisite Review: Conditional Probability", lecture_lines)
        
        # Setup Venn Diagram
        circle_a = Circle(radius=1.2, color=BLUE, fill_opacity=0.3)
        circle_b = Circle(radius=1.2, color=RED, fill_opacity=0.3)
        circle_a.shift(LEFT * 0.6)
        circle_b.shift(RIGHT * 0.6)
        
        venn = VGroup(circle_a, circle_b)
        self.place_in_area(venn, 'B1', 'D4', scale_factor=0.9)
        
        # Intersection area
        intersection = Intersection(circle_a, circle_b, color=YELLOW, fill_opacity=0.6)
        
        # Labels
        label_a = Text("A", font_size=24).next_to(circle_a, UP)
        label_b = Text("B", font_size=24).next_to(circle_b, UP)
        formula = MathTex(r"P(A|B) = \frac{P(A \cap B)}{P(B)}", font_size=32)
        self.place_at_grid(formula, 'E2', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.play(Create(circle_a), Create(circle_b), FadeIn(label_a), FadeIn(label_b))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(GRAY), FadeIn(self.lecture[1]))
        self.play(FadeIn(intersection))
        self.play(Write(formula))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(GRAY), FadeIn(self.lecture[2]))
        self.play(Indicate(circle_b, color=RED), Indicate(intersection, color=YELLOW))
        self.play(FadeOut(venn), FadeOut(intersection), FadeOut(label_a), FadeOut(label_b), FadeOut(formula))
