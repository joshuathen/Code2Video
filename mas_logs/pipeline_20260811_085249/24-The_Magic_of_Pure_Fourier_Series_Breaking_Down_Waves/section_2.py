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
        lecture_lines = [
            "The series formula represents a signal's structure.",
            "The constant term is the average height.",
            "Summing sine waves creates complex oscillations."
        ]
        self.setup_layout("Defining the Pure Fourier Series", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display the general formula for Fourier Series.
        formula = MathTex(
            "f(x) = \\frac{a_0}{2} + \\sum_{n=1}^{\\infty} (a_n \\cos(nx) + b_n \\sin(nx))",
            color=WHITE
        )
        # Applying the fix from issue 37: self.place_in_area(formula, 'B2', 'F5', scale_factor=0.9)
        self.place_in_area(formula, "B2", "F5", scale_factor=0.9)
        self.play(Write(formula))
        self.lecture[0].set_color(WHITE)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight the constant term a0/2.
        # Use sub-mobject indexing to find the a0 term
        a0_term = formula[0][2:7]
        
        self.play(
            Indicate(a0_term, color="#00FFFF"),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Highlight the frequency components (an, bn).
        an_term = formula[0][14:17]
        bn_term = formula[0][22:25]
        
        highlight_freq = VGroup(an_term, bn_term)
        
        self.play(
            highlight_freq.animate.set_color("#FFFF00"),
            self.lecture[2].animate.set_color("#FFFF00")
        )
        self.wait(2)
