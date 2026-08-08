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
        self.setup_layout(
            "The Mathematical Recipe: Defining the Series",
            ["Fourier Series formula decomposes periodic signals.",
             "Coefficients act as weights for each frequency.",
             "Adjusting coefficients shapes the final waveform."]
        )
        
        # Formula setup
        formula = MathTex(
            "f(t) = \\frac{a_0}{2} + \\sum_{n=1}^{\\infty} [a_n \\cos(n\\omega t) + b_n \\sin(n\\omega t)]",
            font_size=36
        )
        
        # Assets
        instrument = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/instrument.svg")
        speaker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speaker.svg")
        
        self.place_at_grid(instrument, 'A6', scale_factor=0.3)
        self.place_at_grid(speaker, 'F6', scale_factor=0.3)
        
        # Position formula per VideoCritic feedback
        self.place_at_grid(formula, 'D4', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        # Fourier Series formula decomposes periodic signals.
        self.play(FadeIn(self.title), FadeIn(instrument), FadeIn(formula))
        self.lecture[0].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Coefficients act as weights for each frequency.
        self.lecture[1].set_color(PURPLE)
        # Highlight coefficients in the formula: a0/2, an, bn
        highlights = VGroup(
            formula[0][3:8],
            formula[0][14:16],
            formula[0][23:25]
        ).set_color(PURPLE)
        self.play(Create(highlights))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Adjusting coefficients shapes the final waveform.
        self.lecture[2].set_color(PINK)
        self.play(FadeIn(speaker))
        # Highlight cosine/sine terms
        cos_part = formula[0][16:22]
        sin_part = formula[0][25:31]
        self.play(
            cos_part.animate.set_color(YELLOW),
            sin_part.animate.set_color(ORANGE),
            Indicate(formula)
        )
        self.wait(2)
