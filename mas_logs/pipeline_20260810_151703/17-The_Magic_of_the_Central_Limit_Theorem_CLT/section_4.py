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
        self.setup_layout("Key Properties & Constraints", [
            "Mean of sample means equals population mean.",
            "Variance decreases as sample size grows.",
            "Independence is essential for this convergence."
        ])
        
        # === Animation for Lecture Line 1 ===
        formula_mean = MathTex(r"\mu_{\bar{x}}", "=", r"\mu", color=WHITE)
        self.place_in_area(formula_mean, 'B2', 'B5', scale_factor=1.0)
        self.play(Write(formula_mean))
        self.play(self.lecture[0].animate.set_color(BLUE))

        # === Animation for Lecture Line 2 ===
        formula_var = MathTex(r"\sigma_{\bar{x}}", "=", r"\frac{\sigma}{\sqrt{n}}", color=WHITE)
        self.place_in_area(formula_var, 'C2', 'C5', scale_factor=1.0)
        self.play(Write(formula_var))
        self.play(self.lecture[1].animate.set_color(GREEN))

        # === Animation for Lecture Line 3 ===
        indep_text = Text("Independence Required", font_size=32, color=YELLOW)
        self.place_at_grid(indep_text, 'D3', scale_factor=0.9)
        
        bell = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bell.svg")
        self.place_at_grid(bell, 'E3', scale_factor=0.5)
        
        self.play(FadeIn(indep_text), FadeIn(bell))
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Bell curve narrowing animation
        self.play(bell.animate.scale(0.5), run_time=1.5)
        self.wait(2)
