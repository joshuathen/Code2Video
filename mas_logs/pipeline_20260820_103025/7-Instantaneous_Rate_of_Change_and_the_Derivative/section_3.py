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
        lecture_lines = [
            "The derivative is the slope of the tangent.",
            "Formula: limit as h approaches zero of slope.",
            "This measures change at an exact point.",
            "Secant slope represents average change over h.",
            "The limit reveals the instantaneous rate."
        ]
        self.setup_layout("Defining the Derivative", lecture_lines)
        
        # Prepare formula
        formula = MathTex(r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}")
        formula.set_color(WHITE)
        # B004/B038: Move formula to 'B2'-'D5' with scale 1.3
        self.place_in_area(formula, "B2", "D5", scale_factor=1.3)

        # Asset: Magnifying Glass
        magnifier = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifyingglass.svg")
        self.place_at_grid(magnifier, "E5", scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        self.play(FadeIn(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF5733")
        num = formula.get_part_by_tex(r"f(x+h) - f(x)")
        self.play(num.animate.set_color("#FF5733"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF5733")
        self.play(FadeIn(magnifier))
        self.play(magnifier.animate.move_to(self.grid["B3"]))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        h_part = formula.get_part_by_tex("h")
        self.play(h_part.animate.set_color("#FF5733"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#33FF57")
        # B028/B038: Move final_symbol to grid F3 with scale 1.5
        final_symbol = MathTex(r"f'(x)")
        self.place_at_grid(final_symbol, "F3", scale_factor=1.5)
        self.play(Flash(final_symbol))
        self.wait(2)
