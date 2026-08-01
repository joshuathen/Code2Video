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
        # Topic: The Bridge Between Change and Accumulation
        # Section 5: FTC Part 2: The Shortcut
        
        self.setup_layout("FTC Part 2: The Shortcut", [
            "We can now evaluate integrals using antiderivatives.",
            "If F is an antiderivative, the area is simple.",
            "Just calculate the difference between the endpoints.",
            "F(b) minus F(a) gives the exact total area.",
            "This shortcut bypasses the need for infinite sums."
        ])

        # Colors
        COLOR_F = "#FF00FF"  # Magenta
        COLOR_f = "#FFA500"  # Orange
        COLOR_MATH = YELLOW

        # Function definitions
        # f(x) = 0.5x + 1
        # F(x) = 0.25x^2 + x + C
        f_func = lambda x: 0.5 * x + 1.0
        F_func = lambda x: 0.25 * x**2 + x + 0.5
        a_val, b_val = 1.0, 4.0

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_MATH)
        
        axes_f = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            x_length=4.0,
            y_length=2.0,
            axis_config={"include_tip": True, "font_size": 20}
        )
        # Resolved Issue 32: Moved to E4-F6 to avoid vertical crowding
        self.place_in_area(axes_f, 'E4', 'F6', scale_factor=0.7)
        
        curve_f = axes_f.plot(f_func, color=COLOR_f)
        label_f = MathTex(r"f(x)", color=COLOR_f).scale(0.7)
        # B027: Update label position to follow axes_f
        self.place_at_grid(label_f, "E6", scale_factor=1.0)
        
        area_f = axes_f.get_area(curve_f, x_range=[a_val, b_val], color=COLOR_f, opacity=0.3)
        
        self.play(Create(axes_f), Create(curve_f), Write(label_f))
        self.play(FadeIn(area_f))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_MATH)
        
        axes_F = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 12, 2],
            x_length=4.0,
            y_length=2.0,
            axis_config={"include_tip": True, "font_size": 20}
        )
        # Resolved Issue 32: Moved to A4-B6 to avoid vertical crowding
        self.place_in_area(axes_F, 'A4', 'B6', scale_factor=0.7)
        
        curve_F = axes_F.plot(F_func, color=COLOR_F)
        label_F = MathTex(r"F(x)", color=COLOR_F).scale(0.7)
        # B027: Update label position to follow axes_F
        self.place_at_grid(label_F, "A6", scale_factor=1.0)
        
        self.play(Create(axes_F), Create(curve_F), Write(label_F))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_MATH)
        
        p_a = axes_F.c2p(a_val, F_func(a_val))
        p_b = axes_F.c2p(b_val, F_func(b_val))
        
        dot_a = Dot(p_a, color=WHITE, radius=0.06)
        dot_b = Dot(p_b, color=WHITE, radius=0.06)
        
        line_a = axes_F.get_vertical_line(p_a, color=WHITE, line_func=DashedLine)
        line_b = axes_F.get_vertical_line(p_b, color=WHITE, line_func=DashedLine)
        
        h_line_a = axes_F.get_horizontal_line(p_a, color=WHITE, line_func=DashedLine)
        h_line_b = axes_F.get_horizontal_line(p_b, color=WHITE, line_func=DashedLine)
        
        label_Fa = MathTex(r"F(a)", color=COLOR_F).scale(0.6)
        label_Fb = MathTex(r"F(b)", color=COLOR_F).scale(0.6)
        
        label_Fa.next_to(axes_F.c2p(0, F_func(a_val)), LEFT, buff=0.1)
        label_Fb.next_to(axes_F.c2p(0, F_func(b_val)), LEFT, buff=0.1)
        
        self.play(Create(line_a), Create(line_b), Create(h_line_a), Create(h_line_b))
        self.play(FadeIn(dot_a), FadeIn(dot_b), Write(label_Fa), Write(label_Fb))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_MATH)
        
        y_start = axes_F.c2p(0, F_func(a_val))
        y_end = axes_F.c2p(0, F_func(b_val))
        
        brace_F = BraceBetweenPoints(y_start, y_end, direction=LEFT, color=COLOR_MATH, buff=0.5)
        brace_text = MathTex(r"F(b)-F(a)", color=COLOR_MATH).scale(0.6).next_to(brace_F, LEFT, buff=0.1)
        
        formula = MathTex(r"\int_a^b f(x) dx = F(b) - F(a)", color=WHITE).scale(0.8)
        # Resolved Issue 30: Centered formula in C2-D5 to avoid overlap with graphs
        self.place_in_area(formula, 'C2', 'D5', scale_factor=0.9)
        
        self.play(GrowFromCenter(brace_F), Write(brace_text))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_MATH)
        
        self.play(
            Indicate(area_f, color=COLOR_MATH),
            Indicate(brace_text, color=COLOR_MATH),
            run_time=2
        )
        
        sum_formula = MathTex(r"\lim_{n \to \infty} \sum_{i=1}^n f(x_i^*) \Delta x", color=GRAY).scale(0.6)
        # Resolved Issue 31: Moved sum_formula to F2 to avoid overlap with lower graph
        self.place_at_grid(sum_formula, 'F2', scale_factor=0.8)
        cross = Cross(sum_formula, color=RED)
        
        self.play(FadeIn(sum_formula))
        self.play(Create(cross))
        self.wait(2)
