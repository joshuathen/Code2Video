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

class Section7ConclusionScene(TeachingScene):
    def construct(self):
        # Initialize the layout with the section title and lecture lines
        self.setup_layout("Conclusion: The Lesson of Moser's Circle", [
            "- Induction helps us guess, but proof confirms truth.",
            "- Always look for the reason behind the pattern.",
            "- In math, proof is the most important ingredient!"
        ])
        
        # === Animation for Lecture Line 1 ===
        # Line: "Induction helps us guess, but proof confirms truth."
        
        # Compare 2^(n-1) and the true formula
        formula_guess = MathTex("2^{n-1}", color=WHITE)
        neq_symbol = MathTex("\\neq", color=RED)
        formula_true = MathTex("\\binom{n}{4} + \\binom{n}{2} + 1", color=WHITE)
        
        comparison_group = VGroup(formula_guess, neq_symbol, formula_true).arrange(RIGHT, buff=0.4)
        # Position per Issue 40: A1-B6, scale 0.8
        self.place_in_area(comparison_group, 'A1', 'B6', scale_factor=0.8)
        
        self.play(Write(comparison_group))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line: "Always look for the reason behind the pattern."
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Display text 'Mathematical Proof' in green
        proof_label = Text("Mathematical Proof", color="#00FF00", font_size=32)
        # Position per Issue 41: C1-C6, scale 0.8
        self.place_in_area(proof_label, 'C1', 'C6', scale_factor=0.8)
        
        self.play(FadeIn(proof_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line: "In math, proof is the most important ingredient!"
        self.play(self.lecture[2].animate.set_color("#FFD700"))
        
        # Integration of assets per Issue 24
        # Max asset
        max_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/max.svg")
        max_icon.set_color("#FFD700")
        
        # Sign asset
        sign_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sign.svg")
        # Text on sign
        sign_text = Text("Proof is the Final Ingredient!", color="#FFD700", font_size=20)
        
        # Position text on sign board
        sign_group = VGroup(sign_icon, sign_text)
        sign_text.move_to(sign_icon.get_center())
        
        # Combine Max and Sign
        conclusion_visual = VGroup(max_icon, sign_group).arrange(UP, buff=0.2)
        # Position per Issue 42: D1-F6, scale 0.8
        self.place_in_area(conclusion_visual, 'D1', 'F6', scale_factor=0.8)
        
        self.play(FadeIn(conclusion_visual))
        self.wait(3)
