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
            "Fourier coefficients act as frequency volume knobs.",
            "They quantify the presence of specific frequencies.",
            "The spectrum grows as we calculate coefficients."
        ]
        self.setup_layout("The Mathematical Framework: Fourier Coefficients", lecture_lines)
        
        # Assets
        knob = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/knob.svg")
        
        # Formula
        formula = MathTex(
            r"f(t) = \frac{a_0}{2} + \sum_{n=1}^{\infty} (a_n \cos(n\omega t) + b_n \sin(n\omega t))",
            font_size=32
        )
        self.place_in_area(formula, 'A2', 'B5', scale_factor=0.9)
        
        # Knob icon next to formula
        knob1 = knob.copy().scale(0.3).next_to(formula, RIGHT, buff=0.3)
        
        # Bar chart placeholder
        bar_chart = VGroup(*[Rectangle(height=1, width=0.3, color=GREEN, fill_opacity=0.6) for _ in range(5)])
        bar_chart.arrange(RIGHT, buff=0.2, aligned_edge=DOWN)
        self.place_in_area(bar_chart, 'C4', 'F6', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(Write(formula), FadeIn(knob1), run_time=2)
        self.lecture[0].set_color("#00FF00")

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(bar_chart), run_time=2)
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        for bar in bar_chart:
            bar.set_height(0)
        
        # Final knob
        knob2 = knob.copy().scale(0.3).move_to(self.grid['F5'])
        
        self.play(
            *[bar.animate.set_height(np.random.uniform(0.5, 2.0)) for bar in bar_chart],
            FadeIn(knob2),
            run_time=3
        )
        self.lecture[2].set_color("#FFA500")
        self.wait(2)
