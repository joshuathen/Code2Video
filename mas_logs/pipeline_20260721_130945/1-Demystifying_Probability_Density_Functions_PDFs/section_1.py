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

class Section1Scene(TeachingScene):
    def construct(self):
        # Fetching context from storyboard and outline
        title_text = "Prerequisite: The Gap Between Discrete and Continuous"
        lecture_lines = [
            "Discrete variables like dice have countable outcomes.",
            "We find probabilities by summing individual values.",
            "Continuous variables like height have infinite possibilities.",
            "The chance of hitting one exact value is zero.",
            "Instead, we measure the probability of ranges."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Hex Colors from storyboard
        GREEN = "#00FF00"
        CYAN = "#00FFFF"
        MAGENTA = "#FF00FF"
        YELLOW = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # "Discrete variables like dice have countable outcomes."
        self.lecture[0].set_color(GREEN)
        
        # Die setup (Procedural as no asset path provided)
        die_square = Square(side_length=1, fill_opacity=1, fill_color=WHITE, color=BLACK)
        die_dots = VGroup(
            Dot(radius=0.08, color=BLACK).move_to(0.25*UP + 0.25*LEFT),
            Dot(radius=0.08, color=BLACK).move_to(0.25*DOWN + 0.25*RIGHT),
            Dot(radius=0.08, color=BLACK).move_to(ORIGIN)
        )
        die = VGroup(die_square, die_dots)
        self.place_at_grid(die, "B2", scale_factor=0.7)
        
        # Simple Bar Chart to represent discrete outcomes
        bars = VGroup(*[
            Rectangle(height=h, width=0.3, fill_opacity=0.8, fill_color=GREEN, stroke_color=WHITE)
            for h in [0.4, 0.9, 1.4, 0.8, 0.5]
        ]).arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
        self.place_in_area(bars, "B3", "D5", scale_factor=0.8)
        
        self.play(FadeIn(die), Write(bars))
        self.play(Rotate(die, angle=PI*2), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "We find probabilities by summing individual values."
        self.lecture[1].set_color(GREEN)
        sum_formula = MathTex(r"P(X) = \sum P(x_i)", color=GREEN)
        self.place_at_grid(sum_formula, "E4", scale_factor=0.8)
        
        self.play(Write(sum_formula))
        # Indicate the summation of the bars
        self.play(LaggedStart(*[Indicate(bar) for bar in bars], lag_ratio=0.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Continuous variables like height have infinite possibilities."
        self.lecture[2].set_color(CYAN)
        
        # Clear previous elements
        self.play(FadeOut(die), FadeOut(bars), FadeOut(sum_formula))
        
        # Horizontal line for continuous axis (e.g., time or height)
        time_line = NumberLine(x_range=[0, 10, 1], length=5, include_numbers=True, color=CYAN)
        self.place_in_area(time_line, "D2", "D6", scale_factor=0.9)
        
        inf_label = Text("Infinite outcomes", font_size=22, color=CYAN)
        # Fix: Issue 18 - Realign with midpoint D4 and avoid clipping
        self.place_at_grid(inf_label, "C4", scale_factor=0.8)
        
        self.play(Create(time_line), Write(inf_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "The chance of hitting one exact value is zero."
        self.lecture[3].set_color(MAGENTA)
        
        # Highlighting a single point on the continuous line
        p_val = 5.0
        p_coord = time_line.number_to_point(p_val)
        dot = Dot(p_coord, color=MAGENTA)
        arrow = Arrow(p_coord + 1.2*UP + 0.4*RIGHT, p_coord, color=MAGENTA, buff=0.1)
        prob_math = MathTex(r"P(X = 5.0) = 0", color=MAGENTA)
        # Fix: Issue 19 - Use area centering for formula
        self.place_in_area(prob_math, 'B3', 'B5', scale_factor=0.8)
        
        self.play(FadeIn(dot), GrowArrow(arrow))
        self.play(Write(prob_math))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Instead, we measure the probability of ranges."
        self.lecture[4].set_color(YELLOW)
        
        # Clean up and transition to smooth curve
        self.play(FadeOut(inf_label), FadeOut(arrow), FadeOut(prob_math), FadeOut(dot))
        
        # Define the PDF function (Gaussian for visualization)
        def pdf_func(x):
            return 1.8 * np.exp(-0.5 * (x - 5)**2 / 1.2**2)
        
        # 1. Blocky Histogram representing high-resolution discrete data
        x_steps = np.linspace(1, 9, 15)
        w = x_steps[1] - x_steps[0]
        blocks = VGroup(*[
            Rectangle(
                height=pdf_func(x), 
                width=w, 
                fill_opacity=0.6, 
                fill_color=GREEN,
                stroke_width=1
            ).move_to(time_line.number_to_point(x), aligned_edge=DOWN)
            for x in x_steps
        ])
        
        # 2. Smooth Curve
        pdf_curve = ParametricFunction(
            lambda t: time_line.number_to_point(t) + pdf_func(t) * UP,
            t_range=[1, 9],
            color=YELLOW
        )
        
        # 3. Shaded Area for range [4, 6]
        pts = [time_line.number_to_point(x) + pdf_func(x) * UP for x in np.linspace(4, 6, 20)]
        pts += [time_line.number_to_point(6), time_line.number_to_point(4)]
        shaded = Polygon(*pts, fill_opacity=0.4, fill_color=YELLOW, stroke_width=0)
        
        range_text = MathTex(r"P(a \le X \le b)", color=YELLOW)
        # Fix: Issue 20 - Use area centering for formula
        self.place_in_area(range_text, 'B3', 'B5', scale_factor=0.8)
        
        # Animation: Morphing the histogram into the curve
        self.play(Create(blocks))
        self.wait(0.5)
        self.play(ReplacementTransform(blocks, pdf_curve))
        self.play(FadeIn(shaded), Write(range_text))
        
        self.wait(2)
