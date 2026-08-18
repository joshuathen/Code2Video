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
            "Derivative is the tangent line's slope.",
            "It measures the instantaneous rate of change.",
            "It is the limit of average rate. [Asset: summary_chart_anim]",
            "Average rate over an interval.",
            "Derivative at a specific point. [Asset: comparison_infographic_anim]"
        ]
        self.setup_layout("Summary and Conclusion", lecture_lines)

        # Reveal lecture lines sequentially
        for line in self.lecture:
            line.set_opacity(1)
            self.wait(0.5)

        # === Animation for Lecture Line 1 ===
        # Derivative is the tangent line's slope.
        self.lecture[0].set_color(BLUE)
        tangent_line = Line(start=LEFT, end=RIGHT, color=BLUE).scale(1.5)
        self.place_in_area(tangent_line, 'A4', 'C6', scale_factor=0.8)
        self.play(Create(tangent_line))

        # === Animation for Lecture Line 2 ===
        # It measures the instantaneous rate of change.
        self.lecture[1].set_color(GREEN)
        rate_text = MathTex(r"\frac{df}{dx}", color=GREEN)
        self.place_at_grid(rate_text, 'B6', scale_factor=1.0)
        self.play(FadeIn(rate_text))

        # === Animation for Lecture Line 3 ===
        # It is the limit of average rate.
        self.lecture[2].set_color(YELLOW)
        limit_text = MathTex(r"\lim_{\Delta x \to 0} \frac{\Delta y}{\Delta x}", color=YELLOW)
        self.place_in_area(limit_text, 'C4', 'D6', scale_factor=1.0)
        self.play(Write(limit_text))

        # === Animation for Lecture Line 4 ===
        # Average rate over an interval.
        self.lecture[3].set_color(ORANGE)
        secant_line = Line(start=LEFT, end=RIGHT, color=ORANGE).scale(1.5).rotate(PI/6)
        self.place_at_grid(secant_line, 'E5', scale_factor=0.8)
        self.play(Create(secant_line))

        # === Animation for Lecture Line 5 ===
        # Derivative at a specific point.
        self.lecture[4].set_color(RED)
        dot = Dot(color=RED)
        self.place_at_grid(dot, 'F6', scale_factor=1.2)
        self.play(GrowFromCenter(dot))

        self.wait(2)
