from manim import *

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

class Section4Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "The Fundamental Link: The Inverse Operation",
            [
                "Integration and differentiation are inverse operations.",
                "They undo each other like addition and subtraction.",
                "The 'Integral Machine' converts velocity into position.",
                "The 'Derivative Machine' converts position back to velocity.",
                "This link is the Fundamental Theorem of Calculus."
            ]
        )
        
        # Colors
        INTEGRAL_COLOR = "#ADD8E6"
        DERIVATIVE_COLOR = "#90EE90"
        BANNER_COLOR = "#FFD700"
        ASSET_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Vertical "Mathematical Mirror"
        mirror_top = self.grid["A3"] + RIGHT * 0.5
        mirror_bottom = self.grid["E3"] + RIGHT * 0.5
        mirror = Line(mirror_top, mirror_bottom, color=WHITE, stroke_width=2)
        mirror_label = Text("Inverse Mirror", font_size=14).next_to(mirror, UP, buff=0.1)
        
        self.play(Create(mirror), Write(mirror_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        plus_symbol = MathTex("+", color=WHITE)
        minus_symbol = MathTex("-", color=WHITE)
        # Fix: Issue 43 and 45: Plus at A2, Minus at A5
        self.place_at_grid(plus_symbol, "A2", scale_factor=1.5)
        self.place_at_grid(minus_symbol, "A5", scale_factor=1.5)
        
        self.play(FadeIn(plus_symbol), FadeIn(minus_symbol))
        self.play(plus_symbol.animate.shift(RIGHT * 0.1), minus_symbol.animate.shift(LEFT * 0.1), rate_func=there_and_back)
        self.wait(1)
        self.play(FadeOut(plus_symbol), FadeOut(minus_symbol))

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(INTEGRAL_COLOR)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg]
        integral_machine = SVGMobject(ASSET_PATH, color=INTEGRAL_COLOR, fill_opacity=1)
        # Fix: Issue 43: Integral machine at B2-C3
        self.place_in_area(integral_machine, "B2", "C3", scale_factor=0.8)
        integral_text = Text("Integral", font_size=12, color=INTEGRAL_COLOR).next_to(integral_machine, DOWN, buff=0.1)
        integral_group = VGroup(integral_machine, integral_text)
        
        v_t = MathTex("v(t)", color=WHITE)
        self.place_at_grid(v_t, "B1")
        
        s_t_out = MathTex("S(t)", color=WHITE)
        self.place_at_grid(s_t_out, "B6")
        
        arrow_in = Arrow(v_t.get_right(), integral_machine.get_left(), buff=0.1, color=INTEGRAL_COLOR)
        arrow_out = Arrow(integral_machine.get_right(), s_t_out.get_left(), buff=0.1, color=INTEGRAL_COLOR)
        
        self.play(FadeIn(integral_group))
        self.play(Write(v_t), GrowArrow(arrow_in))
        self.play(GrowArrow(arrow_out), FadeIn(s_t_out))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(DERIVATIVE_COLOR)
        
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/machine.svg]
        derivative_machine = SVGMobject(ASSET_PATH, color=DERIVATIVE_COLOR, fill_opacity=1)
        # Fix: Issue 44: Derivative machine at D3-E5
        self.place_in_area(derivative_machine, "D3", "E5", scale_factor=0.8)
        derivative_text = Text("Derivative", font_size=12, color=DERIVATIVE_COLOR).next_to(derivative_machine, DOWN, buff=0.1)
        derivative_group = VGroup(derivative_machine, derivative_text)
        
        s_t_in = MathTex("S(t)", color=WHITE)
        self.place_at_grid(s_t_in, "D6")
        
        v_t_out = MathTex("v(t)", color=WHITE)
        self.place_at_grid(v_t_out, "D1")
        
        arrow_back_in = Arrow(s_t_in.get_left(), derivative_machine.get_right(), buff=0.1, color=DERIVATIVE_COLOR)
        arrow_back_out = Arrow(derivative_machine.get_left(), v_t_out.get_right(), buff=0.1, color=DERIVATIVE_COLOR)
        
        self.play(FadeIn(derivative_group))
        self.play(Write(s_t_in), GrowArrow(arrow_back_in))
        self.play(GrowArrow(arrow_back_out), FadeIn(v_t_out))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(BANNER_COLOR)
        
        banner_rect = Rectangle(height=0.6, width=5, color=BANNER_COLOR, fill_opacity=0.5)
        self.place_in_area(banner_rect, "F1", "F6")
        banner_text = Text("Fundamental Theorem of Calculus", font_size=20, color=WHITE).move_to(banner_rect.get_center())
        banner_group = VGroup(banner_rect, banner_text)
        
        self.play(FadeIn(banner_group, shift=UP))
        self.play(banner_group.animate.scale(1.1), rate_func=there_and_back)
        self.wait(2)
        
        # Final color cleanup
        self.lecture[4].set_color(WHITE)
        self.wait(2)
