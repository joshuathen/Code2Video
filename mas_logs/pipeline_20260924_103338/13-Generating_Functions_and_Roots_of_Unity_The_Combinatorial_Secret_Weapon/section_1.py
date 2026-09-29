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
        lecture_lines = [
            "Sequences hide inside power series coefficients.",
            "Think of exponents as weight, coefficients as counts.",
            "A single die becomes this polynomial.",
            "[Asset: DicePolynomial] shows the outcome mapping.",
            "Generating functions are powerful containers for counting."
        ]
        self.setup_layout("Generating Functions as Containers", lecture_lines)
        
        # --- Visual Objects ---
        func_wave = FunctionGraph(lambda x: np.sin(x*3)/2, color="#FFCC00")
        self.place_in_area(func_wave, 'A4', 'B6', scale_factor=0.5)
        
        powers = MathTex("a_0 + a_1x + a_2x^2 + a_3x^3", color="#00FFCC")
        self.place_at_grid(powers, 'C2', scale_factor=0.9)
        
        dice_poly = MathTex("P(x) = x^1 + x^2 + x^3 + x^4 + x^5 + x^6", color="#FFFFFF")
        self.place_at_grid(dice_poly, 'E3', scale_factor=0.8)

        die_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/die.svg", color="#FFFFFF")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(func_wave))
        self.lecture[0].set_color("#FFCC00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Transform(func_wave, powers.copy()))
        self.lecture[1].set_color("#00FFCC")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(dice_poly), FadeIn(self.place_at_grid(die_icon, 'F5', scale_factor=0.5)))
        self.lecture[2].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Represent [Asset: DicePolynomial]
        self.play(Indicate(dice_poly, color="#FF00CC"))
        self.lecture[3].set_color("#FF00CC")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#FFFF00")
        self.wait(2)
