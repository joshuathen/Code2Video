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
        # Ensure LaTeX cleanup doesn't interfere in some environments
        config.no_latex_cleanup = True
        
        lecture_lines = ["Let I sub n be the integral.", "We integrate sine to the power n.", "The area flattens as n increases."]
        self.setup_layout("Prerequisites: The Sine Power Integral", lecture_lines)
        
        # Define mobjects with proper escaping for LaTeX
        i_n_formula = MathTex(r"I_n = \int_0^{\pi/2} \sin^n(x) \, dx", color=WHITE)
        reduction_formula = MathTex(r"I_n = \frac{n-1}{n} I_{n-2}", color=YELLOW)
        
        # --- Animation for Lecture Line 1 ---
        # Fixed alignment based on feedback 20, 22, 35, 37
        self.place_in_area(i_n_formula, 'B3', 'B5', scale_factor=0.8)
        self.play(Write(i_n_formula))
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # --- Animation for Lecture Line 2 ---
        # Selecting the 'n' in \sin^n(x)
        # Note: MathTex indexing is based on the underlying TeX structure. 
        # For I_n = \int_0^{\pi/2} \sin^n(x) dx, index 8 is often 'n'.
        n_highlight = SurroundingRectangle(i_n_formula[0][8], color=RED)
        self.play(Create(n_highlight))
        self.play(self.lecture[1].animate.set_color(RED))
        self.wait(1)

        # --- Animation for Lecture Line 3 ---
        self.play(FadeOut(n_highlight))
        # Fixed alignment based on feedback 21, 22, 36, 37
        self.place_in_area(reduction_formula, 'D3', 'D5', scale_factor=0.8)
        self.play(Write(reduction_formula))
        self.play(self.lecture[2].animate.set_color(YELLOW))
        self.wait(2)
