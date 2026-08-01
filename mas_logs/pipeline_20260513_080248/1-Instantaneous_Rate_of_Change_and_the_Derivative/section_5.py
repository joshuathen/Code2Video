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
        # Setup the layout with lecture lines and title
        lecture_lines_text = [
            'We define the derivative using this formal limit.',
            'The numerator represents the change in vertical output.',
            'The denominator h represents the shrinking horizontal run.',
            'The limit shows h approaching zero precisely.',
            'This resulting value is the derivative, f prime.'
        ]
        self.setup_layout("Formalizing the Derivative", lecture_lines_text)

        # Colors
        RISE_COLOR = "#58D68D"  # Green
        RUN_COLOR = "#F39C12"   # Orange
        FORMULA_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Construction of the limit formula
        lhs = Text("f'(x) =", font_size=24, color=FORMULA_COLOR)
        
        lim_word = Text("lim", font_size=20, color=FORMULA_COLOR)
        h_to_0 = Text("h \u2192 0", font_size=14, color=FORMULA_COLOR)
        limit_unit = VGroup(lim_word, h_to_0).arrange(DOWN, buff=0.05)
        
        num = Text("f(x + h) - f(x)", font_size=20, color=FORMULA_COLOR)
        bar = Line(LEFT, RIGHT, color=FORMULA_COLOR, stroke_width=2).scale(1.2)
        den = Text("h", font_size=20, color=FORMULA_COLOR)
        fraction = VGroup(num, bar, den).arrange(DOWN, buff=0.1)
        
        formula_group = VGroup(lhs, limit_unit, fraction).arrange(RIGHT, buff=0.2)
        # Issue 40 fix: Place formula_group in 'E2' to 'F5'
        self.place_in_area(formula_group, 'E2', 'F5', scale_factor=0.85)

        # Construction of the Graph Area
        axes = Axes(
            x_range=[0, 4, 1],
            y_range=[0, 4, 1],
            x_length=3.5,
            y_length=2.5,
            axis_config={"include_tip": False, "color": GREY_C}
        )
        graph = axes.plot(lambda x: 0.15 * x**2 + 0.5, x_range=[0, 3.5], color=WHITE)
        
        # Initial points for secant
        x0 = 1.0
        h_val = 1.8
        p1 = axes.c2p(x0, 0.15 * x0**2 + 0.5)
        p2 = axes.c2p(x0 + h_val, 0.15 * (x0 + h_val)**2 + 0.5)
        p_corner = axes.c2p(x0 + h_val, 0.15 * x0**2 + 0.5)
        
        dot1 = Dot(p1, radius=0.06, color=WHITE)
        dot2 = Dot(p2, radius=0.06, color=WHITE)
        
        h_line = Line(p1, p_corner, color=RUN_COLOR, stroke_width=4)
        v_line = Line(p_corner, p2, color=RISE_COLOR, stroke_width=4)
        
        graph_group = VGroup(axes, graph, dot1, dot2, h_line, v_line)
        # Issue 39 fix: Place graph_group in 'A2' to 'C5'
        self.place_in_area(graph_group, 'A2', 'C5', scale_factor=0.8)

        # Animation Sequence
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(FadeIn(formula_group), FadeIn(axes), Create(graph))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        self.play(num.animate.set_color(RISE_COLOR))
        self.play(FadeIn(dot1), FadeIn(dot2), Create(v_line))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        self.play(den.animate.set_color(RUN_COLOR))
        self.play(Create(h_line))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        # Limit pulses and h shrinks
        new_h = 0.05
        new_p2 = axes.c2p(x0 + new_h, 0.15 * (x0 + new_h)**2 + 0.5)
        new_p_corner = axes.c2p(x0 + new_h, 0.15 * x0**2 + 0.5)

        self.play(
            limit_unit.animate.scale(1.2).set_color(YELLOW),
            rate_func=there_and_back
        )
        self.play(
            dot2.animate.move_to(new_p2),
            h_line.animate.put_start_and_end_on(p1, new_p_corner),
            v_line.animate.put_start_and_end_on(new_p_corner, new_p2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )

        # Glow and Derivative label
        deriv_label = Text("The Derivative", font_size=20, color=WHITE)
        self.place_at_grid(deriv_label, "D3", scale_factor=1.0)
        
        # Issue 41 fix: Tangent line appears
        # Approximate tangent at x=1 (slope = 2 * 0.15 * 1 = 0.3)
        tangent_line = Line(
            axes.c2p(x0 - 1.5, (0.15 * x0**2 + 0.5) - 0.3 * 1.5),
            axes.c2p(x0 + 1.5, (0.15 * x0**2 + 0.5) + 0.3 * 1.5),
            color=YELLOW, stroke_width=2
        )
        # Using fix from Issue 41 description
        self.place_at_grid(tangent_line, 'B3', scale_factor=0.6)

        self.play(
            formula_group.animate.set_color(WHITE).set_stroke(width=1),
            FadeIn(deriv_label),
            Create(tangent_line)
        )
        self.wait(2)
