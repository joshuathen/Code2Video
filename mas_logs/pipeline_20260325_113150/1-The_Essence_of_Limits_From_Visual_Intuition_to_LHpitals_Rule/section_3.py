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
        # Mandatory call to setup_layout with section title and lecture lines
        self.setup_layout(
            "When Limits Fail: Indeterminate Forms", 
            [
                "Expressions like zero over zero are math stalemates.",
                "Both parts shrink, so we cannot determine the winner.",
                "Standard substitution fails, signaling we need better tools."
            ]
        )

        # Initialize lecture opacity
        for line in self.lecture:
            line.set_opacity(0.3)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_opacity(1))
        
        # Build sin(x)/x limit expression to align with Animation Planner goals
        lim_word = Text("lim", font_size=36)
        sub_text = Text("x → 0", font_size=18).next_to(lim_word, DOWN, buff=0.1)
        lim_part = VGroup(lim_word, sub_text)
        
        num = Text("sin(x)", font_size=36)
        den = Text("x", font_size=36)
        frac_line = Line(LEFT*0.5, RIGHT*0.5, stroke_width=2)
        fraction = VGroup(num, frac_line, den).arrange(DOWN, buff=0.1)
        
        equals = Text("=", font_size=36)
        zero_zero = VGroup(
            Text("0", font_size=36, color=RED),
            Line(LEFT*0.3, RIGHT*0.3, stroke_width=2, color=RED),
            Text("0", font_size=36, color=RED)
        ).arrange(DOWN, buff=0.1)

        expression = VGroup(lim_part, fraction, equals, zero_zero).arrange(RIGHT, buff=0.4)
        
        # Fix Issue 28: Horizontal centering and avoid crowding
        self.place_in_area(expression, 'B2', 'B5', scale_factor=0.9)
        
        self.play(Write(expression[0:2]))
        self.wait(1)
        self.play(Write(expression[2:]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_opacity(0.3),
            self.lecture[1].animate.set_opacity(1)
        )
        
        question_mark = Text("?", font_size=60, color=YELLOW)
        self.place_at_grid(question_mark, "D3")
        
        arrow_num = Arrow(start=expression[1][0].get_top(), end=expression[1][0].get_top()+UP*0.5, color=BLUE)
        arrow_den = Arrow(start=expression[1][2].get_bottom(), end=expression[1][2].get_bottom()+DOWN*0.5, color=BLUE)
        
        shrinking_label = Text("Both vanish", font_size=20, color=BLUE)
        # Fix Issue 30: Better utilization of grid space and visual connection
        self.place_in_area(shrinking_label, 'C3', 'C5', scale_factor=0.7)

        self.play(Create(arrow_num), Create(arrow_den), Write(question_mark), Write(shrinking_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_opacity(0.3),
            self.lecture[2].animate.set_opacity(1)
        )
        
        tool_text = Text("Need L'Hôpital or Algebra", font_size=24, color=GREEN)
        # Fix Issue 29: Center multi-word label
        self.place_in_area(tool_text, 'E2', 'E5', scale_factor=0.8)
        
        self.play(Write(tool_text))
        self.play(Indicate(tool_text))
        self.wait(2)

        # Reset all opacities for conclusion
        self.play(self.lecture.animate.set_opacity(1))
        self.wait(2)
