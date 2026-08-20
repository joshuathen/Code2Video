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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Infinite Sums in 2-adic Space", [
            "Convergence means terms approach zero.", 
            "Terms become increasingly divisible by 2.", 
            "The sum 1+2+4+8 converges to -1."
        ])

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        # Sequence: 1, 2, 4, 8, 16
        powers = VGroup(*[MathTex(f"2^{{{i}}}") for i in range(5)])
        powers.arrange(RIGHT, buff=0.5)
        self.place_in_area(powers, 'A4', 'A6', scale_factor=0.7)
        self.play(Write(powers))
        
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        # Animate terms shifting/changing color
        self.play(*[term.animate.set_color("#ADD8E6") for term in powers])
        self.wait(1)
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # === Animation for Lecture Line 3 ===
        # Show partial sum 1+2+4+8... = -1
        sum_formula = MathTex("1 + 2 + 4 + 8 + \\dots = -1")
        self.place_in_area(sum_formula, 'D4', 'E6', scale_factor=0.8)
        self.play(Write(sum_formula))
        self.wait(2)
