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

class Section3Scene(TeachingScene):
    def construct(self):
        # Setup layout
        lecture_lines = [
            'First, flip the kernel horizontally to prepare.',
            'Slide the kernel across the input signal step-by-step.',
            'Multiply overlapping values at each discrete position.',
            'Sum these products to find the output point.',
            'Repeat across all steps for the full result.'
        ]
        self.setup_layout("The Mechanics: Flip, Slide, Multiply, Sum", lecture_lines)

        # Signal helper for discrete bars
        def create_bar(val, color):
            r = Rectangle(width=0.4, height=val*0.4, fill_opacity=0.8, fill_color=color, stroke_color=color)
            t = Text(str(val), font_size=16)
            return VGroup(r, t).arrange(UP, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(ORANGE)

        # Formula construction
        formula = Text("(x * h)[n] = \u03A3 x[k] * h[n - k]", font_size=24, color=WHITE)
        self.place_in_area(formula, "A1", "A6")
        self.add(formula)

        # Blue Input x: [1, 2, 1]
        x_vals = [1, 2, 1]
        input_bars = VGroup(*[create_bar(v, BLUE) for v in x_vals]).arrange(RIGHT, buff=0.6)
        self.place_in_area(input_bars, "C3", "C5")
        
        # Orange Kernel h: [1, 0.5]
        h_vals = [1, 0.5]
        kernel_bars = VGroup(*[create_bar(v, ORANGE) for v in h_vals]).arrange(RIGHT, buff=0.6)
        self.place_in_area(kernel_bars, "B3", "B4")

        x_label = Text("x[k]", color=BLUE, font_size=20)
        self.place_at_grid(x_label, "C2")
        h_label = Text("h[k]", color=ORANGE, font_size=20)
        self.place_at_grid(h_label, "B2")

        self.play(FadeIn(input_bars), FadeIn(x_label), FadeIn(kernel_bars), FadeIn(h_label))
        self.wait(1)

        # Flipping kernel
        self.play(
            Rotate(kernel_bars, angle=PI, axis=UP),
            FadeOut(h_label),
            run_time=1.5
        )
        # Correct text mirroring after rotation
        for bar in kernel_bars:
            bar[1].rotate(PI, axis=UP)
            
        h_flipped_label = Text("h[n-k]", color=ORANGE, font_size=20)
        self.place_at_grid(h_flipped_label, "B2", scale_factor=0.8)
        self.play(FadeIn(h_flipped_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(ORANGE)
        
        # Position flipped kernel so the original '1' (now at visual right) aligns with first input
        target_x = input_bars[0].get_x()
        current_x = kernel_bars[0].get_x()
        self.play(kernel_bars.animate.shift(RIGHT * (target_x - current_x)))
        self.wait(1)

        # === Animation for Lecture Lines 3, 4, and 5 ===
        steps = [
            {"n": 0, "calc": "1 * 1 = 1", "res": 1, "bars": lambda k, x: [k[0], x[0]], "grid": "E3"},
            {"n": 1, "calc": "1*2 + 0.5*1 = 2.5", "res": 2.5, "bars": lambda k, x: [k[0], x[1], k[1], x[0]], "grid": "E4"},
            {"n": 2, "calc": "1*1 + 0.5*2 = 2.0", "res": 2.0, "bars": lambda k, x: [k[0], x[2], k[1], x[1]], "grid": "E5"},
            {"n": 3, "calc": "0.5 * 1 = 0.5", "res": 0.5, "bars": lambda k, x: [k[1], x[2]], "grid": "E6"}
        ]
        
        y_label = Text("y[n]", color=TEAL, font_size=20)
        self.place_at_grid(y_label, "E2")
        self.play(FadeIn(y_label))

        step_width = input_bars[1].get_x() - input_bars[0].get_x()

        for i, step in enumerate(steps):
            if i == 0:
                self.lecture[2].set_color(YELLOW)
            elif i == 1:
                self.lecture[3].set_color(TEAL)
                self.lecture[4].set_color(TEAL)
            
            if i > 0:
                self.play(kernel_bars.animate.shift(RIGHT * step_width), run_time=1.0)
            
            overlap_objs = step["bars"](kernel_bars, input_bars)
            highlight = SurroundingRectangle(VGroup(*overlap_objs), color=YELLOW, buff=0.1)
            calc_text = Text(step["calc"], color=YELLOW, font_size=20)
            self.place_at_grid(calc_text, "D4", scale_factor=0.9)
            
            self.play(Create(highlight), FadeIn(calc_text))
            self.wait(0.5)
            
            out_bar = create_bar(step["res"], TEAL)
            self.place_at_grid(out_bar, step["grid"])
            
            self.play(ReplacementTransform(calc_text.copy(), out_bar))
            self.play(FadeOut(highlight), FadeOut(calc_text), run_time=0.5)
            
        self.wait(2)
