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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Deriving the Product Formula", [
            "Substitute the reduction formula into the ratio.",
            "This yields a pattern for pi over two.",
            "We arrange these terms into an infinite product."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Substitute the reduction formula into the ratio.
        # Write recursive formula I_n = (n-1)/n * I_{n-2}.
        formula = MathTex(r"I_n = \frac{n-1}{n} I_{n-2}", color=WHITE)
        self.place_at_grid(formula, 'B3', scale_factor=0.8)
        self.play(Write(formula))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Apply recursion step visually for I_n.
        # This yields a pattern for pi over two.
        formula2 = MathTex(r"\frac{I_{2n+1}}{I_{2n}} = \prod_{k=1}^{n} \frac{(2k)^2}{(2k-1)(2k+1)}", color="#FFCC00")
        self.place_at_grid(formula2, 'D2', scale_factor=0.7)
        self.play(Transform(formula.copy(), formula2))
        self.play(self.lecture[1].animate.set_color("#FFCC00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # We arrange these terms into an infinite product.
        final_form = MathTex(r"\frac{\pi}{2} = \prod_{k=1}^{\infty} \frac{(2k)^2}{(2k-1)(2k+1)}", color="#FF6600")
        self.place_in_area(final_form, 'D3', 'F5', scale_factor=0.9)
        self.play(FadeIn(final_form))
        self.play(self.lecture[2].animate.set_color("#00FFCC"))
        
        box = SurroundingRectangle(final_form, color=WHITE, buff=0.1)
        self.play(Create(box))
        self.play(Flash(box, color=WHITE))
        self.wait(2)
