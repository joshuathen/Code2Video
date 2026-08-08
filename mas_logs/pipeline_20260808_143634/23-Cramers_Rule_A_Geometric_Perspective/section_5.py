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
        self.setup_layout("Summary & Intuition", ["Cramer's rule measures target space.", "Relates b to basis vectors.", "Geometrically intuitive scaling factors."])
        
        # === Animation for Lecture Line 1 ===
        # Summarize Cramer's rule as area ratios.
        ratio_text = MathTex(r"\text{Ratio} = \frac{\text{Area}_{i}}{\text{Area}_{\text{base}}}", color=YELLOW)
        self.place_at_grid(ratio_text, "B3", scale_factor=1.0)
        self.play(Write(ratio_text))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display the formula x_i = det(A_i) / det(A).
        formula = MathTex(r"x_i = \frac{\det(A_i)}{\det(A)}", color=BLUE)
        self.place_in_area(formula, "C4", "C5", scale_factor=1.2)
        self.play(FadeIn(formula))
        self.lecture[1].set_color(BLUE)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Geometrically intuitive scaling factors.
        scaling_text = Text("Geometric Intuition", color=GREEN, font_size=24)
        self.place_in_area(scaling_text, "E3", "E5", scale_factor=0.9)
        self.play(Write(scaling_text))
        self.lecture[2].set_color(GREEN)
        self.wait(1)
        
        # Final fade out
        self.play(FadeOut(ratio_text), FadeOut(formula), FadeOut(scaling_text), FadeOut(self.lecture), FadeOut(self.title))
        self.wait(1)
