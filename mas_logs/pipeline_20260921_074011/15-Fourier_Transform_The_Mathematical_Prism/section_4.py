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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The transform multiplies signal by sine waves.",
            "Correlation results highlight frequency presence.",
            "Matching waves isolate signal energy.",
            "Non-matching frequencies cancel out.",
            "This process sifts hidden components."
        ]
        self.setup_layout("The Fourier Transform Integral", lecture_lines)
        
        # Prepare Formula
        formula = MathTex(r"F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-i\omega t} dt", font_size=36)
        
        # Load Assets
        radio = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/radio.svg")
        speaker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speaker.svg")
        
        # VideoCritic requested positioning
        self.place_in_area(formula, 'B2', 'C5', scale_factor=0.9)
        self.place_at_grid(radio, 'A5', scale_factor=0.5)
        self.place_at_grid(speaker, 'F5', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Write(formula), FadeIn(radio))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        # Highlighting 't' in the formula
        t_part = formula[0][8]
        self.play(t_part.animate.set_color("#FFFF00"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        # Highlighting complex exponential
        exp_part = formula[0][10:15]
        self.play(exp_part.animate.set_color("#00FF00"))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FF8800"))
        # Integral sign and limits
        int_part = formula[0][3:6]
        self.play(int_part.animate.set_color("#FF8800"))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF00FF"))
        # Whole result part
        res_part = formula[0][0:2]
        self.play(res_part.animate.set_color("#FF00FF"), FadeIn(speaker))
        
        self.wait(2)
