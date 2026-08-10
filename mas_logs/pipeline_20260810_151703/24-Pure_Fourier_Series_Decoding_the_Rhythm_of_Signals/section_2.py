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
        lecture_lines = [
            "Fourier series decomposes signals into harmonic components.",
            "Coefficients determine the weight of each component.",
            "Watch the square wave form, piece by piece.",
            "Harmonics build the signal's unique profile.",
            "Integration reveals the amplitude of every frequency."
        ]
        self.setup_layout("Mathematical Framework: The Harmonic Foundation", lecture_lines)
        
        # Assets
        synth_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/synthesizer.svg")
        scope_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/oscilloscope.svg")
        
        # Formula setup - B029: fix formula positioning
        formula = MathTex(
            r"f(t) = \sum_{n=-\infty}^{\infty} c_n e^{in\omega t}",
            font_size=32, color="#00FF00"
        )
        self.place_in_area(formula, 'B2', 'C5', scale_factor=0.9)
        
        # B030: harmonics dots group
        harmonics_dots = VGroup(*[Dot(color=BLUE) for _ in range(5)])
        self.place_in_area(harmonics_dots, 'D2', 'E5', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#00FF00"))
        self.place_at_grid(synth_icon, 'B1', scale_factor=0.3)
        self.play(FadeIn(formula), FadeIn(synth_icon))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        self.play(Indicate(formula))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        self.play(FadeIn(harmonics_dots))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF00FF"))
        self.play(harmonics_dots.animate.set_color(RED))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFA500"))
        self.place_at_grid(scope_icon, 'F6', scale_factor=0.3)
        self.play(FadeIn(scope_icon))
