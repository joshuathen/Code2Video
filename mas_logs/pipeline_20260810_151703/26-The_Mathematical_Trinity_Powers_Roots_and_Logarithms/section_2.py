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
        self.setup_layout("The Inverse: Roots (Uncovering the Base)", [
            "Roots uncover the original base.",
            "Imagine a square of nine sunflowers.",
            "The square root of nine is three."
        ])
        
        # === Animation for Lecture Line 1 ===
        root_formula = MathTex(r"\\sqrt{y} = b", color=WHITE)
        self.place_at_grid(root_formula, "B3", scale_factor=1.2)
        self.play(Write(root_formula))
        self.play(self.lecture[0].animate.set_color(YELLOW))

        # === Animation for Lecture Line 2 ===
        sunflowers = VGroup(*[Dot(color=YELLOW) for _ in range(9)]).arrange_in_grid(3, 3, buff=0.1)
        self.place_in_area(sunflowers, "D2", "F3", scale_factor=0.8)
        self.play(FadeIn(sunflowers))
        self.play(self.lecture[1].animate.set_color(GREEN))

        # === Animation for Lecture Line 3 ===
        sqrt_nine = MathTex(r"\\sqrt{9} = 3", color="#F44336")
        self.place_at_grid(sqrt_nine, "B5", scale_factor=1.0)
        
        # Collapse and split visual
        self.play(
            FadeOut(root_formula),
            ReplacementTransform(sunflowers, sqrt_nine)
        )
        self.play(self.lecture[2].animate.set_color("#F44336"))
        self.wait(2)
