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
        title = "Testing the Concept: Polynomials as Vectors"
        lines = ["Let's treat polynomials like vectors.", "Add coefficients just like grid coordinates.", "The internal structure is identical."]
        self.setup_layout(title, lines)
        
        # Define objects
        poly = MathTex("x^2 + x", color=WHITE)
        coords = MathTex(r"\\begin{bmatrix} 1 \\\\ 1 \\\\ 0 \\end{bmatrix}", color="#1abc9c")
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        arrow = Arrow(start=UP, end=DOWN, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#1abc9c")
        self.place_in_area(poly, "B1", "B3", scale_factor=1.2)
        self.play(FadeIn(poly))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#1abc9c")
        self.place_at_grid(grid_asset, "C3", scale_factor=0.6)
        self.place_at_grid(coords, "C4", scale_factor=0.8)
        self.play(FadeIn(grid_asset), FadeIn(coords))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#1abc9c")
        arrow.next_to(poly, DOWN, buff=0.2)
        self.play(Create(arrow))
        self.wait(2)
