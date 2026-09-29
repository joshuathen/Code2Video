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
        self.setup_layout("Prerequisite: The Harmonic Series", [
            "Consider the harmonic series sum 1/n.",
            "When s=1, the series diverges.",
            "Increasing s makes terms shrink faster."
        ])
        
        # === Animation for Lecture Line 1 ===
        series = MathTex(r"1 + \frac{1}{2} + \frac{1}{3} + \dots", color=BLUE)
        self.place_in_area(series, 'B3', 'B5', scale_factor=0.7)
        self.play(Write(series))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        sum_text = MathTex(r"\sum_{n=1}^{\infty} \frac{1}{n} = \infty", color=YELLOW)
        self.place_at_grid(sum_text, 'C3', scale_factor=0.9)
        self.play(FadeIn(sum_text))
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        compare_text = MathTex(r"\frac{1}{n} \to \frac{1}{n^s}, s > 1", color=GREEN)
        self.place_at_grid(compare_text, 'D3', scale_factor=0.8)
        self.play(Write(compare_text))
        self.lecture[2].set_color(GREEN)
        
        self.wait(2)
