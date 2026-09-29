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
        self.setup_layout("The Core Concept: The Fourier Series Formula", [
            "The Fourier series is an infinite harmonic sum.", 
            "Coefficients act like volume knobs for frequencies.", 
            "Adding harmonics gradually sharpens the signal shape."
        ])

        # Assets
        knob_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/knob.svg"
        knob = SVGMobject(knob_path)
        
        # Formula construction
        formula = MathTex(
            "f(t) = \\frac{a_0}{2} + \\sum_{n=1}^{\\infty} \\left( a_n \\cos(n \\omega t) + b_n \\sin(n \\omega t) \\right)",
            font_size=32
        )
        # Applying the fix from VideoCritic (Issue 42): place in C2-E6
        self.place_in_area(formula, 'C2', 'E6', scale_factor=0.95)

        # === Animation for Lecture Line 1 ===
        # The Fourier series is an infinite harmonic sum.
        self.play(FadeIn(formula))
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        # Coefficients act like volume knobs for frequencies.
        an_bn_highlight = formula[0][8:10] # a_n
        bn_highlight = formula[0][17:19]  # b_n
        self.place_at_grid(knob, "A2", scale_factor=0.5)
        self.play(
            FadeIn(knob),
            Indicate(an_bn_highlight, color="#FF00FF"),
            Indicate(bn_highlight, color="#FF00FF"),
            self.lecture[1].animate.set_color("#FF00FF")
        )

        # === Animation for Lecture Line 3 ===
        # Highlight summation symbol
        sum_highlight = formula[0][7:8] # sum
        self.play(
            Indicate(sum_highlight, color="#00FFFF"),
            self.lecture[2].animate.set_color("#00FFFF")
        )

        # Animate harmonic term
        harmonic_highlight = formula[0][11:16] # cos(...)
        self.play(
            Indicate(harmonic_highlight, color="#FFFF00"),
            self.lecture[2].animate.set_color("#FFFF00")
        )

        # Animate wave sharpening with knob
        self.play(
            knob.animate.rotate(2 * PI),
            self.lecture[2].animate.set_color("#00FF00")
        )
        self.wait(2)
