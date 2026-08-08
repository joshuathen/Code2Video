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
        self.setup_layout("The Mathematical Construction", [
            "The Fourier series constructs complex periodic signals.",
            "Each term acts as a weighted harmonic.",
            "a₀ represents the signal's average height.",
            "aₙ and bₙ are volume knobs for frequencies.",
            "Adjusting coefficients reshapes the final function."
        ])
        
        # Formula elements
        formula = MathTex(
            r"f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos(nx) + b_n \sin(nx))"
        )
        # Adjusted formula position per issue 27/42
        self.place_in_area(formula, 'C2', 'E5', scale_factor=0.9)
        
        # Asset for knob
        knob_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/knob.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(Write(formula))
        self.play(self.lecture[0].animate.set_color("#00FFFF"))

        # === Animation for Lecture Line 2 ===
        # Highlight coefficients and use asset
        an = formula.get_parts_by_tex("a_n")
        bn = formula.get_parts_by_tex("b_n")
        weights = VGroup(an, bn)
        
        knob = self.place_at_grid(knob_icon.copy(), 'D4', scale_factor=0.5)
        self.play(weights.animate.set_color("#FF00FF"), FadeIn(knob))
        self.play(self.lecture[1].animate.set_color("#FF00FF"))

        # === Animation for Lecture Line 3 ===
        a0 = formula.get_parts_by_tex("a_0")
        self.play(a0.animate.set_color("#00FF00"))
        self.play(self.lecture[2].animate.set_color("#00FF00"))

        # === Animation for Lecture Line 4 ===
        # Re-highlight coefficients as knobs
        self.play(weights.animate.set_color("#FF8800"))
        self.play(self.lecture[3].animate.set_color("#FF8800"))

        # === Animation for Lecture Line 5 ===
        # Final color shift to show full interaction
        self.play(formula.animate.set_color(WHITE))
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.wait(2)
