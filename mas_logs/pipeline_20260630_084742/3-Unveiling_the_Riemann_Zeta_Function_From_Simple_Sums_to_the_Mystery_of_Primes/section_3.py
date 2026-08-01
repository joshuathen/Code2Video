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
        # Initialization
        lines = [
            "The series fails to converge when 's' is small.",
            "At 's' equals one, the harmonic sum grows infinitely.",
            "We visualize 's' as points on a complex grid."
        ]
        self.setup_layout("The Boundary: Convergence vs. Divergence", lines)
        
        # Colors
        COLOR_GREEN = "#00FF00"
        COLOR_RED = "#FF0000"
        COLOR_WHITE = "#FFFFFF"
        COLOR_HIGHLIGHT = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(COLOR_HIGHLIGHT))
        
        # Formula - Using Text instead of MathTex to avoid LaTeX dependency
        formula = Text("Σ 1/nˢ", color=COLOR_WHITE)
        self.place_at_grid(formula, "C4", scale_factor=0.9)
        
        series_expansion = Text("1 + 1/2 + 1/3 + ...", color=COLOR_WHITE)
        self.place_at_grid(series_expansion, "D4", scale_factor=0.5)
        
        self.play(Write(formula))
        self.play(FadeIn(series_expansion))

        # Rising green blocks (initial ones)
        stack = VGroup()
        base_y = self.grid["F3"][1]
        center_x = self.grid["F3"][0]
        
        block_widths = [1.2, 1.0, 0.8, 0.7, 0.6]
        block_heights = [1.0, 0.5, 0.33, 0.25, 0.2]
        
        current_y = base_y
        for w, h in zip(block_widths, block_heights):
            rect = Rectangle(width=w, height=h, fill_opacity=0.8, fill_color=COLOR_GREEN, stroke_color=WHITE, stroke_width=1)
            rect.move_to([center_x, current_y + h/2, 0])
            stack.add(rect)
            current_y += h
            
        self.play(Create(stack, lag_ratio=0.3), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(COLOR_WHITE),
            self.lecture[1].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        extra_blocks = VGroup()
        for i in range(6, 25):
            h = 1.0/i
            w = 0.5
            rect = Rectangle(width=w, height=h, fill_opacity=0.8, fill_color=COLOR_GREEN, stroke_color=WHITE, stroke_width=0.5)
            rect.move_to([center_x, current_y + h/2, 0])
            extra_blocks.add(rect)
            current_y += h
            
        self.play(Create(extra_blocks, lag_ratio=0.1), run_time=2)
        
        self.play(
            VGroup(stack, extra_blocks).animate.shift(UP * 4),
            series_expansion.animate.set_opacity(0),
            formula.animate.set_opacity(0),
            run_time=2
        )
        self.wait(1)
        
        self.play(FadeOut(stack), FadeOut(extra_blocks))

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(COLOR_WHITE),
            self.lecture[2].animate.set_color(COLOR_HIGHLIGHT)
        )
        
        # Complex Grid - Use label_constructor=Text to avoid LaTeX dependency
        plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_numbers": True, "font_size": 18, "label_constructor": Text}
        )
        self.place_in_area(plane, "A1", "F6", scale_factor=0.9)
        
        self.play(Create(plane))
        
        # Critical point s=1
        s_point = Dot(plane.c2p(1, 0), color=COLOR_RED)
        s_label = Text("s=1", color=COLOR_RED, font_size=24)
        s_label.next_to(s_point, UP + RIGHT, buff=0.1)
        
        # Red flashing warning
        warning_box = Rectangle(width=1.5, height=0.6, color=COLOR_RED, stroke_width=2)
        warning_box.move_to(plane.c2p(1, 0))
        warning_text = Text("DIVERGENCE!", color=COLOR_RED, font_size=18).move_to(warning_box.get_center())
        warning_group = VGroup(warning_box, warning_text)
        self.place_at_grid(warning_group, "A6", scale_factor=1.0)
        
        self.play(FadeIn(s_point), Write(s_label))
        
        for _ in range(3):
            self.play(Indicate(warning_group, color=COLOR_RED, scale_factor=1.2), run_time=0.5)
            self.play(warning_group.animate.set_opacity(0.3), run_time=0.2)
            self.play(warning_group.animate.set_opacity(1), run_time=0.2)
            
        self.wait(2)
