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
        self.setup_layout("The 2-adic Lens: Changing the Rules", [
            "2-adic valuation: size based on divisibility by two.",
            "Many factors of two make numbers small.",
            "See 16 shrink closer to zero than 2.",
            "A magnifying glass reveals this hidden scale.",
            "Numbers with more powers of two feel smaller."
        ])
        
        # Define mobjects
        val_symbol = MathTex(r"|x|_2", font_size=60, color=WHITE)
        factorization = MathTex(r"16 = 2^4", font_size=40, color="#FFFF00")
        comparison = MathTex(r"|16|_2 < |2|_2", font_size=40, color="#00FFFF")
        
        # Asset: Magnifier icon
        magnifier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifier.svg", color="#00FFFF")

        # === Animation for Lecture Line 1 ===
        self.place_at_grid(val_symbol, 'B4', scale_factor=1.0)
        self.play(FadeIn(val_symbol))
        self.lecture[0].set_color(WHITE)

        # === Animation for Lecture Line 2 ===
        self.place_in_area(factorization, 'D2', 'D2', scale_factor=0.8)
        self.play(FadeIn(factorization))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(comparison, 'D4', scale_factor=1.0)
        self.play(FadeIn(comparison))
        self.lecture[2].set_color("#00FFFF")

        # === Animation for Lecture Line 4 ===
        self.place_at_grid(magnifier, 'C5', scale_factor=0.9)
        self.play(Create(magnifier))
        self.lecture[3].set_color("#00FFFF")

        # === Animation for Lecture Line 5 ===
        self.play(Indicate(factorization), Indicate(comparison))
        self.lecture[4].set_color(ORANGE)
        
        self.wait(2)
