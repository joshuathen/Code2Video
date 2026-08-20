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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Defining Independence", [
            "Independence means knowledge of B doesn't change A.",
            "Two events are independent if P(A|B) equals P(A).",
            "Independent events share no predictive information."
        ])
        
        # Visual assets
        circle_a = Circle(color=BLUE, fill_opacity=0.3).set_stroke(width=2)
        circle_b = Circle(color=RED, fill_opacity=0.3).set_stroke(width=2)
        label_a = MathTex("A").next_to(circle_a, UP)
        label_b = MathTex("B").next_to(circle_b, UP)
        group = VGroup(circle_a, label_a, circle_b, label_b)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_opacity(1)
        self.place_in_area(group, 'A2', 'A5', scale_factor=0.9)
        self.play(FadeIn(group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFF00") # Light yellow
        self.lecture[1].set_opacity(1)
        
        formula = MathTex("P(A|B) = P(A)").scale(0.8)
        self.place_in_area(formula, 'B2', 'B5', scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#32CD32") # Lime Green
        self.lecture[2].set_opacity(1)
        
        formula_2 = MathTex("P(A \\cap B) = P(A) \\cdot P(B)").scale(0.8)
        formula_2.set_color("#32CD32")
        self.place_in_area(formula_2, 'D2', 'D5', scale_factor=0.8)
        self.play(ReplacementTransform(formula, formula_2))
        self.wait(2)
