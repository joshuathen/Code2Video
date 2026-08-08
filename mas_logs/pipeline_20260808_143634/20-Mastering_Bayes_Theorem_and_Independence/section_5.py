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
        self.setup_layout("Synthesis & Summary", [
            "Independence simplifies, while Bayes' updates probabilities.",
            "Bayes' is the heart of modern machine learning.",
            "Spam filters use Bayes' to classify emails."
        ])
        
        # Prep assets
        formula_indep = MathTex(r"P(A|B) = P(A)", color=BLUE)
        formula_bayes = MathTex(r"P(A|B) = \frac{P(B|A)P(A)}{P(B)}", color=YELLOW)
        
        summary_box = Rectangle(width=5, height=2, color=WHITE)
        summary_text = Text("Updating Beliefs with Evidence", font_size=24, color=GREEN)
        summary_group = VGroup(summary_box, summary_text).arrange(DOWN)
        
        # Applying critique: Fix summary group position
        self.place_in_area(summary_group, 'A3', 'B5', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(Write(formula_indep), run_time=1)
        self.place_at_grid(formula_indep, 'A2')
        self.play(FadeOut(formula_indep))
        
        # Applying critique: Fix formula Bayes position
        self.play(Write(formula_bayes), run_time=1)
        self.place_in_area(formula_bayes, 'B3', 'C5', scale_factor=0.8)
        self.play(FadeOut(formula_bayes))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.play(Create(summary_box))
        self.play(Write(summary_text))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        # Using asset per instructions
        email_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/email.svg")
        # Applying critique: Fix email icon position
        self.place_at_grid(email_icon, 'F5', scale_factor=0.9)
        self.play(FadeIn(email_icon))
        self.wait(2)
