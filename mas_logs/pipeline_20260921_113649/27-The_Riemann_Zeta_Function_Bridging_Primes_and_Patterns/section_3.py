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
        lines = ["Zeta connects to prime numbers.", "A product formula reveals this.", "Primes act like fundamental building blocks.", "Each prime contributes a factor.", "They define the Zeta structure."]
        self.setup_layout("The Euler Product Formula", lines)
        
        # Asset integration
        bricks = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bricks.svg")
        
        # Animation Elements
        zeta_eq = MathTex(r"\zeta(s) = \prod_{p} (1 - p^{-s})^{-1}").scale(1.2)
        prime_group = VGroup(*[Text(str(p), color="#9B59B6") for p in [2, 3, 5, 7, 11]]).arrange(RIGHT)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#F1C40F"))
        self.place_at_grid(bricks, 'A1', 0.5)
        self.play(FadeIn(bricks))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#F1C40F"))
        self.place_at_grid(prime_group, 'A3', 1.0)
        self.play(FadeIn(prime_group))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#F1C40F"))
        self.place_in_area(zeta_eq, 'B2', 'E5', 0.9)
        self.play(Write(zeta_eq))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#F1C40F"))
        zeta_eq.set_color("#9B59B6")
        self.play(Indicate(zeta_eq))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#F1C40F"))
        bracket = Brace(zeta_eq, DOWN, color="#34495E")
        self.place_in_area(bracket, 'D2', 'F5', 0.8)
        self.play(Create(bracket))
        self.place_at_grid(bricks, 'F6', 0.5)
        self.play(FadeIn(bricks))
