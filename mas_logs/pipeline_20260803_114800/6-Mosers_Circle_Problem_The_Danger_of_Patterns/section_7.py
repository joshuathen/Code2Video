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

class Section7Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion: The Lesson of Moser", [
            "Patterns in math can be misleading.",
            "Never rely on induction without a formal proof.",
            "Always check the next case!"
        ])
        
        # === Animation for Lecture Line 1 ===
        # Line 1: Patterns in math can be misleading.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        proof_text = Text("Math requires rigorous proof", font_size=32, color="#FFFFFF")
        self.place_in_area(proof_text, 'A1', 'A6', scale_factor=0.8)
        self.play(Write(proof_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: Never rely on induction without a formal proof.
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        misleading_text = Text("Induction can be misleading", font_size=32, color="#FF0000")
        self.place_in_area(misleading_text, 'C1', 'C6', scale_factor=0.8)
        self.play(FadeIn(misleading_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: Always check the next case!
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        formula_label = Text("Moser's Formula:", font_size=24, color=WHITE)
        self.place_in_area(formula_label, 'D1', 'D6', scale_factor=0.7)
        
        formula = MathTex(r"\binom{n}{4} + \binom{n}{2} + 1", font_size=42, color=BLUE_B)
        self.place_in_area(formula, 'E1', 'F6', scale_factor=0.9)
        
        self.play(Write(formula_label), Write(formula))
        self.wait(2)
        
        # Final cleanup
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(3)
