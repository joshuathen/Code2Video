from manim import *
import numpy as np

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
        # Setup the layout with lecture lines
        lecture_lines = [
            "Let's solve the determinant for this specific matrix.", 
            "Expanding the determinant yields a characteristic quadratic equation.", 
            "Solving this equation reveals eigenvalues of five and two."
        ]
        self.setup_layout("Step-by-Step Calculation Example", lecture_lines)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        
        # Matrix A = [[4, 1], [2, 3]] in white (#FFFFFF)
        m_4 = Text("4", font_size=36, color=WHITE)
        m_1 = Text("1", font_size=36, color=WHITE)
        m_2 = Text("2", font_size=36, color=WHITE)
        m_3 = Text("3", font_size=36, color=WHITE)
        
        vals_row1 = VGroup(m_4, m_1).arrange(RIGHT, buff=1.2)
        vals_row2 = VGroup(m_2, m_3).arrange(RIGHT, buff=1.2)
        matrix_vals = VGroup(vals_row1, vals_row2).arrange(DOWN, buff=0.8)
        
        l_bracket = Text("[", font_size=100, color=WHITE)
        r_bracket = Text("]", font_size=100, color=WHITE)
        matrix_full = VGroup(l_bracket, matrix_vals, r_bracket).arrange(RIGHT, buff=0.2)
        
        a_label = Text("A =", font_size=36, color=WHITE)
        matrix_disp = VGroup(a_label, matrix_full).arrange(RIGHT, buff=0.3)
        
        # Shifted up to Row A-B (Issue 36)
        self.place_in_area(matrix_disp, "A1", "B6")
        
        self.play(FadeIn(matrix_disp))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        
        # Animate diagonal subtractions: [[4-λ, 1], [2, 3-λ]] in gold (#FFD700)
        l_bar = Text("|", font_size=100, color="#FFD700")
        r_bar = Text("|", font_size=100, color="#FFD700")
        m_4_sub = Text("4 - \u03BB", font_size=30, color="#FFD700")
        m_3_sub = Text("3 - \u03BB", font_size=30, color="#FFD700")
        m_1_copy = Text("1", font_size=30, color="#FFD700")
        m_2_copy = Text("2", font_size=30, color="#FFD700")
        
        det_vals_row1 = VGroup(m_4_sub, m_1_copy).arrange(RIGHT, buff=0.8)
        det_vals_row2 = VGroup(m_2_copy, m_3_sub).arrange(RIGHT, buff=0.8)
        det_vals = VGroup(det_vals_row1, det_vals_row2).arrange(DOWN, buff=0.8)
        
        det_full = VGroup(l_bar, det_vals, r_bar).arrange(RIGHT, buff=0.2)
        det_eq = Text("= 0", font_size=36, color="#FFD700")
        det_disp = VGroup(det_full, det_eq).arrange(RIGHT, buff=0.3)
        
        # Position same area as matrix (Issue 36)
        self.place_in_area(det_disp, "A1", "B6")
        
        self.play(ReplacementTransform(matrix_disp, det_disp))
        self.wait(1)

        # Expand determinant: (4-λ)(3-λ) - 2 = 0 in yellow (#FFFF00)
        expansion_text = Text("(4 - \u03BB)(3 - \u03BB) - 2 = 0", font_size=28, color="#FFFF00")
        self.place_in_area(expansion_text, "C1", "C6") # Issue 37
        
        self.play(Write(expansion_text))
        self.wait(1)

        # Simplify to characteristic polynomial: λ² - 7λ + 10 = 0 in yellow (#FFFF00)
        quad_text = Text("\u03BB\u00B2 - 7\u03BB + 10 = 0", font_size=32, color="#FFFF00")
        self.place_in_area(quad_text, "C1", "C6") # Replacing expansion_text at same row (Issue 37)
        
        self.play(ReplacementTransform(expansion_text, quad_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GREEN)
        
        # Factoring: (λ - 5)(λ - 2) = 0
        factor_text = Text("(\u03BB - 5)(\u03BB - 2) = 0", font_size=32, color="#FFFF00")
        self.place_in_area(factor_text, "D1", "D6") # Issue 37
        
        self.play(FadeIn(factor_text, shift=DOWN*0.1))
        self.wait(1)

        # Final result: λ₁ = 5, λ₂ = 2 in green (#00FF00)
        result_text = Text("\u03BB\u2081 = 5,  \u03BB\u2082 = 2", font_size=36, color="#00FF00")
        self.place_in_area(result_text, "E1", "E6") # Issue 38
        
        # Asset integration (Issue 26)
        icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/based.svg")
        icon.scale(0.3)
        icon.next_to(result_text, RIGHT, buff=0.5)
        
        final_group = VGroup(result_text, icon)
        
        self.play(Write(result_text))
        self.play(FadeIn(icon))
        
        # Pulse the result and asset (Issue 26)
        self.play(
            final_group.animate.scale(1.2),
            rate_func=there_and_back,
            run_time=0.8
        )
        self.wait(2)
