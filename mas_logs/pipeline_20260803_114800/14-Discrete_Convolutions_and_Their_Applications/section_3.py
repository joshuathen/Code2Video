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
        lecture_lines = [
            "First, we flip the filter h horizontally.",
            "Next, slide the flipped filter along the signal.",
            "Multiply the overlapping values at each step.",
            "Sum these products to find the output value.",
            "Repeat for every position to get the result."
        ]
        self.setup_layout("The Math: Flip, Slide, Multiply, Add", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Display the formula (x * h)[n] = \sum x[k]h[n-k] in gold (#FFD700).
        formula = MathTex(r"(x * h)[n] = \sum_{k=-\infty}^{\infty} x[k]h[n-k]", color="#FFD700")
        # Issue 23 Fix: Place formula in A4-A6
        self.place_in_area(formula, 'A4', 'A6', scale_factor=0.7)
        self.add(formula)

        # Show signal x[k] as blue bars (#0000FF) and filter h[k] as red bars (#FF0000).
        x_vals = [1.0, 2.0, 1.5]
        x_bars = VGroup(*[
            Rectangle(width=0.4, height=v*0.5, fill_opacity=0.8, fill_color="#0000FF", stroke_color=WHITE)
            for v in x_vals
        ]).arrange(RIGHT, buff=0.6)
        x_labels = VGroup(*[MathTex(str(v), font_size=18) for v in x_vals])
        for i, bar in enumerate(x_bars):
            x_labels[i].next_to(bar, UP, buff=0.1)
        signal_group = VGroup(x_bars, x_labels)
        self.place_in_area(signal_group, "C2", "C4", scale_factor=1.0)
        
        x_name = Text("x[k]", font_size=18, color="#0000FF")
        self.place_at_grid(x_name, "C1", scale_factor=1.0)
        self.add(signal_group, x_name)

        h_vals = [0.5, 1.0]
        h_bars = VGroup(*[
            Rectangle(width=0.4, height=v*0.5, fill_opacity=0.8, fill_color="#FF0000", stroke_color=WHITE)
            for v in h_vals
        ]).arrange(RIGHT, buff=0.6)
        h_labels = VGroup(*[MathTex(str(v), font_size=18) for v in h_vals])
        for i, bar in enumerate(h_bars):
            h_labels[i].next_to(bar, UP, buff=0.1)
        filter_group = VGroup(h_bars, h_labels)
        self.place_in_area(filter_group, "B3", "B4", scale_factor=1.0)
        
        h_name = Text("h[k]", font_size=18, color="#FF0000")
        self.place_at_grid(h_name, "B2", scale_factor=1.0)
        self.add(filter_group, h_name)

        # Flip h[k] horizontally to become h[-k] with a 'Flip' label.
        self.lecture[0].set_color(YELLOW)
        h_flipped_vals = [1.0, 0.5]
        h_flipped_bars = VGroup(*[
            Rectangle(width=0.4, height=v*0.5, fill_opacity=0.8, fill_color="#FF0000", stroke_color=WHITE)
            for v in h_flipped_vals
        ]).arrange(RIGHT, buff=0.6)
        h_flipped_labels = VGroup(*[MathTex(str(v), font_size=18) for v in h_flipped_vals])
        for i, bar in enumerate(h_flipped_bars):
            h_flipped_labels[i].next_to(bar, UP, buff=0.1)
        flipped_group = VGroup(h_flipped_bars, h_flipped_labels)
        
        # Issue 24 Fix: Place flipped_group in D2-D3 and flip_label in D1
        self.place_in_area(flipped_group, "D2", "D3", scale_factor=1.0)
        
        flip_label = Text("h[-k]", font_size=18, color=WHITE)
        self.place_at_grid(flip_label, "D1", scale_factor=1.0)
        
        self.play(ReplacementTransform(filter_group.copy(), flipped_group), Write(flip_label))
        self.wait(1)

        # === Animation for Lecture Line 2 & 3 ===
        # Next, slide the flipped filter along the signal.
        # Multiply the overlapping values at each step.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.lecture[2].set_color(YELLOW)

        # Initial alignment for n=0
        target_center = x_bars[0].get_center()
        # The "active" part of flipped filter is h[0] which is the 2nd element (h_flipped_bars[1])
        shift_vec = target_center - h_flipped_bars[1].get_center()
        self.play(flipped_group.animate.shift(shift_vec))
        
        # Flash overlap
        flash_rect = Rectangle(width=0.5, height=1.0, color=YELLOW, fill_opacity=0.3).move_to(x_bars[0])
        self.play(Flash(x_bars[0], color=YELLOW), FadeIn(flash_rect))
        self.play(FadeOut(flash_rect))

        # === Animation for Lecture Line 4 ===
        # Sum these products to find the output value.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Issue 25 Fix: Place y_name in F2 and y0_bar in F3
        y_name = Text("y[n]", font_size=18, color="#00FF00")
        self.place_at_grid(y_name, "F2", scale_factor=1.0)
        self.add(y_name)

        # n=0 result
        y0_val = 1.0 * 0.5
        y0_bar = Rectangle(width=0.4, height=y0_val*0.5, fill_opacity=0.8, fill_color="#00FF00", stroke_color=WHITE)
        self.place_at_grid(y0_bar, "F3", scale_factor=1.0)
        y0_lbl = MathTex("0.5", font_size=18).next_to(y0_bar, UP, buff=0.1)
        self.play(FadeIn(y0_bar), FadeIn(y0_lbl))

        # === Animation for Lecture Line 5 ===
        # Repeat for every position to get the result.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # n=1
        self.play(flipped_group.animate.shift(RIGHT * 1.0))
        # Flash overlap for n=1
        flash_rect1 = Rectangle(width=1.5, height=1.0, color=YELLOW, fill_opacity=0.3).move_to(VGroup(x_bars[0], x_bars[1]))
        self.play(Flash(VGroup(x_bars[0], x_bars[1]), color=YELLOW), FadeIn(flash_rect1))
        self.play(FadeOut(flash_rect1))

        y1_val = (1.0 * 1.0) + (2.0 * 0.5)
        y1_bar = Rectangle(width=0.4, height=y1_val*0.5, fill_opacity=0.8, fill_color="#00FF00", stroke_color=WHITE)
        self.place_at_grid(y1_bar, "F4", scale_factor=1.0)
        y1_lbl = MathTex("2.0", font_size=18).next_to(y1_bar, UP, buff=0.1)
        self.play(FadeIn(y1_bar), FadeIn(y1_lbl))

        # n=2
        self.play(flipped_group.animate.shift(RIGHT * 1.0))
        # Flash overlap for n=2
        flash_rect2 = Rectangle(width=1.5, height=1.0, color=YELLOW, fill_opacity=0.3).move_to(VGroup(x_bars[1], x_bars[2]))
        self.play(Flash(VGroup(x_bars[1], x_bars[2]), color=YELLOW), FadeIn(flash_rect2))
        self.play(FadeOut(flash_rect2))

        y2_val = (2.0 * 1.0) + (1.5 * 0.5)
        y2_bar = Rectangle(width=0.4, height=y2_val*0.5, fill_opacity=0.8, fill_color="#00FF00", stroke_color=WHITE)
        self.place_at_grid(y2_bar, "F5", scale_factor=1.0)
        y2_lbl = MathTex("2.75", font_size=18).next_to(y2_bar, UP, buff=0.1)
        self.play(FadeIn(y2_bar), FadeIn(y2_lbl))

        # n=3
        self.play(flipped_group.animate.shift(RIGHT * 1.0))
        # Flash overlap for n=3
        flash_rect3 = Rectangle(width=0.5, height=1.0, color=YELLOW, fill_opacity=0.3).move_to(x_bars[2])
        self.play(Flash(x_bars[2], color=YELLOW), FadeIn(flash_rect3))
        self.play(FadeOut(flash_rect3))

        y3_val = (1.5 * 1.0)
        y3_bar = Rectangle(width=0.4, height=y3_val*0.5, fill_opacity=0.8, fill_color="#00FF00", stroke_color=WHITE)
        self.place_at_grid(y3_bar, "F6", scale_factor=1.0)
        y3_lbl = MathTex("1.5", font_size=18).next_to(y3_bar, UP, buff=0.1)
        self.play(FadeIn(y3_bar), FadeIn(y3_lbl))

        self.wait(2)
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
