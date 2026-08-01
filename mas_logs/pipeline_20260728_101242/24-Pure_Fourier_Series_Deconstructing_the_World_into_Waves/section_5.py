from manim import *
import numpy as np

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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "- Let's build a square wave using only odd harmonics.",
            "- Start with the fundamental sine wave at base frequency.",
            "- Add the third harmonic at one-third of the amplitude.",
            "- Higher odd harmonics sharpen the corners of the shape.",
            "- Ripples at the edges are known as Gibbs Phenomenon."
        ]
        self.setup_layout("Case Study: Assembling the Square Wave", lecture_lines)

        # Colors
        COLOR_1 = WHITE
        COLOR_2 = "#FF0000" # Red
        COLOR_3 = "#00FF00" # Green
        COLOR_4 = "#0000FF" # Blue
        COLOR_5 = "#FFFF00" # Yellow

        # === Animation for Lecture Line 1 ===
        # - Let's build a square wave using only odd harmonics.
        self.lecture[0].set_color(COLOR_1)
        
        axes = Axes(
            x_range=[0, 2 * PI, PI],
            y_range=[-1.5, 1.5, 1],
            x_length=5,
            y_length=3,
            axis_config={"include_tip": False}
        )
        self.place_in_area(axes, "B1", "F6")
        
        self.play(Create(axes))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # - Start with the fundamental sine wave at base frequency.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_2)
        
        func_1 = axes.plot(lambda x: np.sin(x), color=COLOR_2)
        label_1 = MathTex(r"\sin(x)", color=COLOR_2, font_size=28)
        self.place_in_area(label_1, "A1", "A6") # Fix: issue 31
        
        self.play(Create(func_1), Write(label_1))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # - Add the third harmonic at one-third of the amplitude.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_3)
        
        harmonic_3 = axes.plot(lambda x: (1/3) * np.sin(3*x), color=COLOR_3, stroke_opacity=0.4)
        sum_2 = axes.plot(lambda x: np.sin(x) + (1/3) * np.sin(3*x), color=COLOR_3)
        label_2 = MathTex(r"\sin(x) + \frac{1}{3}\sin(3x)", color=COLOR_3, font_size=28)
        self.place_in_area(label_2, "A1", "A6") # Fix: issue 31
        
        self.play(
            Create(harmonic_3),
            FadeOut(label_1),
            run_time=1
        )
        self.play(
            Transform(func_1, sum_2),
            Write(label_2),
            FadeOut(harmonic_3),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # - Higher odd harmonics sharpen the corners of the shape.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(COLOR_4)
        
        def get_square_approx(x, n_terms):
            res = 0
            for k in range(1, n_terms + 1):
                n = 2 * k - 1
                res += (1/n) * np.sin(n * x)
            return res

        label_4 = MathTex(r"S_N(x) = \sum_{k=1}^{N} \frac{\sin((2k-1)x)}{2k-1}", color=COLOR_4, font_size=28)
        self.place_in_area(label_4, "A1", "A4") # Fix: issue 32
        
        self.play(FadeOut(label_2), Write(label_4))
        
        # Sequentially add harmonics up to N=10
        for n_terms in range(3, 11):
            new_sum = axes.plot(lambda x, nt=n_terms: get_square_approx(x, nt), color=COLOR_4)
            self.play(Transform(func_1, new_sum), run_time=0.4)
        
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # - Ripples at the edges are known as Gibbs Phenomenon.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(COLOR_5)
        
        # Gibbs overshoot circles
        c1 = Circle(radius=0.1, color=COLOR_5).move_to(axes.c2p(0.15, 1.1))
        c2 = Circle(radius=0.1, color=COLOR_5).move_to(axes.c2p(PI-0.15, 1.1))
        c3 = Circle(radius=0.1, color=COLOR_5).move_to(axes.c2p(PI+0.15, -1.1))
        c4 = Circle(radius=0.1, color=COLOR_5).move_to(axes.c2p(2*PI-0.15, -1.1))
        
        label_5 = Text("Gibbs Phenomenon", color=COLOR_5, font_size=24)
        self.place_in_area(label_5, "A5", "A6") # Fix: issue 33
        
        self.play(
            Create(VGroup(c1, c2, c3, c4)),
            Write(label_5)
        )
        self.wait(3)
