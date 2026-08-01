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

class Section5Scene(TeachingScene):
    def create_fraction(self, num_text, den_text, color=WHITE, font_size=24):
        n = Text(num_text, font_size=font_size, color=color)
        d = Text(den_text, font_size=font_size, color=color)
        line = Line(LEFT * 0.4, RIGHT * 0.4, color=color, stroke_width=2)
        line.width = max(n.width, d.width) + 0.1
        res = VGroup(n, line, d).arrange(DOWN, buff=0.1)
        return res

    def construct(self):
        lecture_lines = [
            "To minimize time, set the derivative to zero.",
            "Differentiate the time function with respect to x.",
            "Use the chain rule for the square roots.",
            "Set the simplified expression equal to zero.",
            "This equation defines the fastest possible path."
        ]
        self.setup_layout("Optimization via Calculus", lecture_lines)

        # Colors
        YELLOW_OPT = "#FFFF00"
        BLUE_TERM = "#00CCFF"
        GREEN_TERM = "#00FF00"
        WHITE_CLR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW_OPT)
        opt_cond = Text("dT / dx = 0", font_size=36, color=YELLOW_OPT, weight=BOLD)
        # Issue 41 Fix: move to area B2-B5 to align with future final_eq
        self.place_in_area(opt_cond, "B2", "B5", scale_factor=1.2)
        self.play(Write(opt_cond))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE_CLR)
        self.lecture[1].set_color(BLUE_TERM)
        
        # Show derivation of first term
        term1_deriv_label = Text("d/dx (d1/v1) =", font_size=24, color=BLUE_TERM)
        self.place_at_grid(term1_deriv_label, "C2", scale_factor=0.9)
        
        frac1 = self.create_fraction("x", "v1 · d1", color=BLUE_TERM)
        # Issue 42 Fix: move to C3
        self.place_at_grid(frac1, "C3", scale_factor=1.0)
        
        self.play(FadeIn(term1_deriv_label), FadeIn(frac1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE_CLR)
        self.lecture[2].set_color(GREEN_TERM)
        
        # Show derivation of second term
        term2_deriv_label = Text("d/dx (d2/v2) =", font_size=24, color=GREEN_TERM)
        self.place_at_grid(term2_deriv_label, "D2", scale_factor=0.9)
        
        frac2 = self.create_fraction("-(w - x)", "v2 · d2", color=GREEN_TERM)
        # Issue 43 Fix: move to D3
        self.place_at_grid(frac2, "D3", scale_factor=1.0)
        
        self.play(FadeIn(term2_deriv_label), FadeIn(frac2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE_CLR)
        self.lecture[3].set_color(YELLOW_OPT)
        
        # Combine terms: x/(v1 d1) - (w-x)/(v2 d2) = 0
        eq_part1 = self.create_fraction("x", "v1 · d1", color=BLUE_TERM)
        minus_sign = Text("-", font_size=30, color=WHITE_CLR)
        eq_part2 = self.create_fraction("w - x", "v2 · d2", color=GREEN_TERM)
        equals_zero = Text("= 0", font_size=30, color=YELLOW_OPT)
        
        final_eq = VGroup(eq_part1, minus_sign, eq_part2, equals_zero).arrange(RIGHT, buff=0.3)
        self.place_in_area(final_eq, "B2", "B5", scale_factor=0.8)
        
        self.play(
            FadeOut(opt_cond),
            FadeOut(term1_deriv_label),
            FadeOut(term2_deriv_label),
            ReplacementTransform(frac1, eq_part1),
            ReplacementTransform(frac2, eq_part2),
            FadeIn(minus_sign),
            FadeIn(equals_zero)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE_CLR)
        self.lecture[4].set_color(WHITE_CLR)
        
        # Asset Loading for Issue 29
        axis_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/axis.svg")
        graph_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/graph.svg")
        self.place_at_grid(axis_icon, "F5", scale_factor=0.4)
        self.place_at_grid(graph_icon, "E1", scale_factor=0.4)

        # Visualization: Graph of T(x)
        axes = Axes(
            x_range=[0, 6, 1],
            y_range=[0, 5, 1],
            axis_config={"include_tip": False, "font_size": 18},
            x_length=3,
            y_length=2
        )
        x_label = Text("x", font_size=18).next_to(axes.x_axis, RIGHT, buff=0.1)
        t_label = Text("T(x)", font_size=18).next_to(axes.y_axis, UP, buff=0.1)
        graph_group = VGroup(axes, x_label, t_label)
        self.place_in_area(graph_group, "E2", "F5", scale_factor=1.0)
        
        curve = axes.plot(lambda x: 0.5 * (x - 3)**2 + 1, x_range=[1, 5], color=YELLOW)
        
        # Dot for sliding x
        dot_x = Dot(axes.c2p(1, 0), color=WHITE)
        # Dot on curve
        dot_t = Dot(axes.c2p(1, 3), color=YELLOW_OPT)
        # Connecting line (Simplified for performance, use single updater)
        trace_line = Line(dot_x.get_center(), dot_t.get_center(), color=GREY_A, stroke_width=2)
        
        self.play(
            Create(axes), FadeIn(x_label), FadeIn(t_label),
            FadeIn(axis_icon), FadeIn(graph_icon)
        )
        self.play(Create(curve))
        self.add(trace_line, dot_x, dot_t)
        
        # Animation: sliding x and tracking T(x)
        x_tracker = ValueTracker(1)
        
        def update_dots(m):
            val = x_tracker.get_value()
            dot_x.move_to(axes.c2p(val, 0))
            dot_t.move_to(axes.c2p(val, 0.5 * (val - 3)**2 + 1))
            trace_line.put_start_and_end_on(dot_x.get_center(), dot_t.get_center())

        dot_x.add_updater(update_dots)
        
        self.play(x_tracker.animate.set_value(3), run_time=2, rate_func=linear)
        
        min_label = Text("Minimum", font_size=16, color=YELLOW_OPT)
        min_arrow = Arrow(start=UP, end=DOWN, color=YELLOW_OPT, buff=0.1).scale(0.3)
        min_vgroup = VGroup(min_label, min_arrow).arrange(DOWN, buff=0.1)
        min_vgroup.next_to(dot_t, UP, buff=0.1)
        
        self.play(FadeIn(min_vgroup))
        self.play(x_tracker.animate.set_value(5), run_time=1.5, rate_func=linear)
        
        dot_x.remove_updater(update_dots)
        self.wait(2)
