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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Conclusion: The Need for Rigorous Proof", [
            "Obvious patterns can sometimes lead to false conclusions.",
            "Mathematical rigor is necessary to prove any hypothesis.",
            "Always verify your results with a formal proof."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Display the pattern 2^{n-1} and place a large red 'X' over it (Cross, #FF0000).
        self.lecture[0].set_color(YELLOW)
        seq_formula = MathTex("2^{n-1}", color=WHITE)
        # Fix from Issue 36: Adjusted position and scale for visual balance
        self.place_in_area(seq_formula, 'A3', 'B4', scale_factor=1.4)
        
        cross = Cross(seq_formula, stroke_color="#FF0000", stroke_width=12)
        
        self.play(Write(seq_formula))
        self.play(Create(cross))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Show the combination formula with a green checkmark (Check, #00FF00).
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        true_formula = MathTex(r"R = \binom{n}{4} + \binom{n}{2} + 1", color=WHITE)
        # Fix from Issue 37: Moved up to reduce gap and adjusted scale
        self.place_in_area(true_formula, 'C2', 'D5', scale_factor=1.3)
        
        # Green checkmark
        check = MathTex(r"\checkmark", color="#00FF00").scale(1.5)
        check.next_to(true_formula, RIGHT, buff=0.4)
        
        self.play(Write(true_formula))
        self.play(FadeIn(check))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Fade in the text 'Deductive Proof > Inductive Reasoning' (Text, #FFFF00).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        proof_text = Text("Deductive Proof > Inductive Reasoning", font_size=24, color="#FFFF00")
        # Fix from Issue 38: Centered the text and restricted width to prevent overlap
        self.place_in_area(proof_text, 'E2', 'E5', scale_factor=0.8)
        
        self.play(FadeIn(proof_text))
        self.wait(2)
