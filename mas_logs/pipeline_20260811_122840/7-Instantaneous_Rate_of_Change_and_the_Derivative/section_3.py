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
            "Derivative: lim h->0 [f(x+h)-f(x)]/h.",
            "Rise is f(x+h)-f(x), Run is h.",
            "As h shrinks, average rate becomes instantaneous.",
            "Example: f(x)=x^2 gives f'(x)=2x.",
            "This is our mathematical tool for change."
        ]
        self.setup_layout("The Mathematical Definition", lecture_lines)
        self.lecture.set_opacity(1)

        # Colors for lecture lines
        colors = ["#FFFFFF", "#00CED1", "#FF6347", "#FFFF00", "#7FFF00"]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(colors[0])
        formula = MathTex(r"f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}", color=colors[0])
        # Applied constraint from issue #36 and #47
        self.place_in_area(formula, 'B4', 'B6', scale_factor=0.7)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(colors[1])
        rise_run = VGroup(
            MathTex(r"\text{Rise} = f(x+h) - f(x)", color=colors[1]),
            MathTex(r"\text{Run} = h", color=colors[1])
        ).arrange(DOWN)
        # Applied constraint from issue #37
        self.place_at_grid(rise_run, 'C5', scale_factor=0.6)
        self.play(FadeIn(rise_run))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(colors[2])
        # Visualize h shrinking
        h_line = Line(start=self.grid['D4'], end=self.grid['D6'], color=colors[2])
        self.play(Create(h_line), run_time=1)
        self.play(h_line.animate.scale(0.3), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(colors[3])
        example = MathTex(r"f(x)=x^2 \implies f'(x)=2x", color=colors[3])
        # Applied constraint from issue #38
        self.place_at_grid(example, 'D5', scale_factor=0.7)
        self.play(Write(example))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(colors[4])
        # Highlighting the formula utility as requested in storyboard
        highlight_box = SurroundingRectangle(formula, color=colors[4], buff=0.2)
        self.play(Create(highlight_box))
        self.wait(2)
