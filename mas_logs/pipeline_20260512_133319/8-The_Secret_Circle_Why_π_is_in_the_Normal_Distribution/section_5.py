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
        # Initialization
        lecture_lines = [
            'This extra r enables a simple substitution.', 
            'Integrating across the radius yields exactly one half.', 
            'Now we sweep through a full circle of rotation.', 
            'Integrating through two-pi radians gives us pi.', 
            'The hidden circle finally reveals its mathematical constant.'
        ]
        self.setup_layout("The 'Aha!' Moment: The Integral Solved", lecture_lines)

        # Helper to build integral parts
        def create_integral_display(lower_t, upper_t, color=WHITE):
            sym = Text("\u222b", font_size=40, color=color)
            low = Text(lower_t, font_size=16, color=color)
            up = Text(upper_t, font_size=16, color=color)
            low.next_to(sym, DOWN, buff=-0.1).shift(RIGHT*0.1)
            up.next_to(sym, UP, buff=-0.1).shift(RIGHT*0.1)
            return VGroup(sym, low, up)

        # Colors
        COLOR_WHITE = "#FFFFFF"
        COLOR_GREEN = "#83C167"
        COLOR_BLUE = "#58C4DD"
        COLOR_YELLOW = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_WHITE)
        
        int1 = create_integral_display("0", "2\u03c0", color=COLOR_WHITE)
        int2 = create_integral_display("0", "\u221e", color=COLOR_WHITE)
        integrand = Text("e\u207b\u02b3\u00b2 r dr d\u03b8", font_size=28, color=COLOR_WHITE)
        
        polar_integral = VGroup(int1, int2, integrand).arrange(RIGHT, buff=0.2)
        # Issue 40: Center the polar integral in a smaller area A2-A5
        self.place_in_area(polar_integral, 'A2', 'A5', scale_factor=1.0)
        
        u_sub = Text("u = r\u00b2  \u21d2  du = 2r dr", font_size=24, color=COLOR_WHITE)
        u_sub_rearrange = Text("\u00bd du = r dr", font_size=24, color=COLOR_WHITE)
        u_group = VGroup(u_sub, u_sub_rearrange).arrange(DOWN, aligned_edge=LEFT)
        # Issue 39: Move u-substitution steps to B2-C5 to save space
        self.place_in_area(u_group, 'B2', 'C5', scale_factor=0.8)
        
        r_dr_highlight = Ellipse(width=0.8, height=0.4, color=COLOR_YELLOW, stroke_width=2)
        r_dr_highlight.move_to(integrand).shift(RIGHT * 0.4)
        
        self.play(FadeIn(polar_integral))
        self.wait(1)
        self.play(FadeIn(u_group))
        self.play(Create(r_dr_highlight))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_GREEN)
        
        int_eval = create_integral_display("0", "2\u03c0", color=COLOR_GREEN)
        inner_result = Text("\u00bd d\u03b8", font_size=28, color=COLOR_GREEN)
        simplified_integral = VGroup(int_eval, inner_result).arrange(RIGHT, buff=0.2)
        
        final_eq_sign = Text(" =", font_size=28, color=COLOR_YELLOW)
        final_val = Text("\u03c0", font_size=28, color=COLOR_YELLOW)
        final_result_group = VGroup(final_eq_sign, final_val).arrange(RIGHT, buff=0.1)
        
        equation_row = VGroup(simplified_integral, final_result_group).arrange(RIGHT, buff=0.5)
        # Issue 38: Place simplified integral at row D to avoid overlap with u-sub
        self.place_in_area(equation_row, 'D1', 'D6', scale_factor=0.9)
        final_result_group.set_opacity(0)
        
        self.play(
            FadeOut(u_group),
            FadeOut(r_dr_highlight),
            ReplacementTransform(polar_integral.copy(), simplified_integral)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_BLUE)
        
        center_pt = self.grid['E3']
        circle_outline = Circle(radius=0.9, color=WHITE, stroke_width=2)
        self.place_at_grid(circle_outline, 'E3', scale_factor=1.0)
        
        sweeper_arm = Line(center_pt, center_pt + RIGHT * 0.9, color=COLOR_BLUE, stroke_width=4)
        
        fill_sector = AnnularSector(
            inner_radius=0, 
            outer_radius=0.9, 
            angle=TAU, 
            start_angle=0, 
            fill_opacity=0.4, 
            color=COLOR_BLUE,
            arc_center=center_pt
        )
        
        self.play(Create(circle_outline))
        self.play(Create(sweeper_arm))
        self.play(
            Rotate(sweeper_arm, angle=TAU, about_point=center_pt),
            Create(fill_sector),
            run_time=3,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_YELLOW)
        self.play(final_result_group.animate.set_opacity(1))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_YELLOW)
        
        self.play(
            equation_row.animate.scale(1.15),
            circle_outline.animate.set_color(COLOR_YELLOW)
        )
        self.wait(2)
