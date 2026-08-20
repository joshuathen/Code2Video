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
            "The Fourier series builds waves from components.",
            "The constant a0 sets the average height.",
            "Coefficients an and bn define harmonic volume.",
            "Adding sine waves approximates complex shapes.",
            "Summing more harmonics creates a perfect square."
        ]
        self.setup_layout("The Mathematical Formula: Building Blocks", lecture_lines)
        
        # Assets
        wave_icon1 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wave.svg")
        wave_icon2 = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/wave.svg")

        # Formula setup
        formula = MathTex(
            "f(t) = \\frac{a_0}{2} + \\sum_{n=1}^{\\infty} [a_n \\cos(n\\omega t) + b_n \\sin(n\\omega t)]",
            font_size=36
        )
        self.place_in_area(formula, 'B2', 'B5', scale_factor=1.0)
        self.place_at_grid(wave_icon1, 'A5', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Write(formula), FadeIn(wave_icon1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF1493"))
        self.play(Indicate(formula[0][5:9], color="#FF1493"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00BFFF"))
        self.play(Indicate(formula[0][16:21], color="#00BFFF"), Indicate(formula[0][24:29], color="#00BFFF"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#ADFF2F"))
        sine = FunctionGraph(lambda x: np.sin(x), x_range=[-PI, PI])
        self.place_at_grid(sine, 'D4', scale_factor=0.6)
        self.play(Create(sine))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FFD700"))
        square = FunctionGraph(lambda x: 1 if x > 0 else -1, x_range=[-PI, PI])
        self.place_at_grid(square, 'E4', scale_factor=0.6)
        self.place_at_grid(wave_icon2, 'F5', scale_factor=0.5)
        self.play(Transform(sine, square), FadeIn(wave_icon2))
        self.wait(1)
