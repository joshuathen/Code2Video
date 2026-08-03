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
            "The geometric series formula works for 2-adic numbers.",
            "Plugging in values yields a result of negative one.",
            "Partial sums get closer to negative one eventually.",
            "It resembles an eight-bit computer's overflow error.",
            "The sum wraps around to land on negative one."
        ]
        self.setup_layout("Calculating the Impossible: Why it equals -1", lecture_lines)
        
        # Colors
        formula_color = "#DA70D6" # Orchid
        sums_color = "#87CEEB"    # Sky Blue
        overflow_color = "#FFD700" # Gold
        merge_color = "#ADFF2F"    # Green Yellow

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(formula_color)
        formula = MathTex(r"S = \frac{a}{1 - r}", color=formula_color)
        self.place_in_area(formula, "A1", "B3", scale_factor=1.0)
        
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(formula_color)
        formula_eval = MathTex(r"S = \frac{1}{1 - 2} = -1", color=formula_color)
        self.place_in_area(formula_eval, "A4", "B6", scale_factor=1.0)
        
        self.play(Write(formula_eval))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(sums_color)
        
        # Show partial sums: 1, 3, 7, 15
        # [ISSUE 37] Move sums list to C1-D2 to optimize space
        sums = VGroup(
            MathTex("S_0 = 1", color=sums_color),
            MathTex("S_1 = 3", color=sums_color),
            MathTex("S_2 = 7", color=sums_color),
            MathTex("S_3 = 15", color=sums_color)
        ).arrange(DOWN, aligned_edge=LEFT)
        self.place_in_area(sums, "C1", "D2", scale_factor=0.8)
        
        for s in sums:
            self.play(FadeIn(s, shift=RIGHT))
            self.wait(0.2)
            
        # Distance calculation d_2(15, -1) = 1/16
        # [ISSUE 36] Expand horizontal area to C3-D6 for better readability
        dist_calc = MathTex(r"d_2(15, -1) = |15 - (-1)|_2", color=sums_color)
        dist_calc_2 = MathTex(r"= |16|_2 = \frac{1}{16}", color=sums_color)
        dist_group = VGroup(dist_calc, dist_calc_2).arrange(DOWN)
        self.place_in_area(dist_group, "C3", "D6", scale_factor=0.8)
        
        self.play(Write(dist_group))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(overflow_color)
        
        # Circular overflow visualization
        circle = Circle(radius=0.7, color=overflow_color)
        self.place_in_area(circle, "E1", "F3", scale_factor=1.0)
        
        # [ISSUE 38] Refine circle label placement using grid anchors
        label_0 = Text("0 (00...0)", font_size=16, color=WHITE)
        self.place_at_grid(label_0, "E2", scale_factor=1.0)
        label_0.shift(UP * 0.6) # Fine tuning above circle
        
        label_max = Text("255 (11...1)", font_size=16, color=WHITE)
        self.place_at_grid(label_max, "E1", scale_factor=1.0)
        label_max.shift(RIGHT * 0.2 + UP * 0.2)
        
        # Overflow arrow
        overflow_arrow = CurvedArrow(
            circle.point_at_angle(PI/2 + 0.4), 
            circle.point_at_angle(PI/2 - 0.4), 
            color=RED,
            radius=0.9
        )
        
        self.play(Create(circle))
        self.play(FadeIn(label_0), FadeIn(label_max))
        self.play(Create(overflow_arrow))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(merge_color)
        
        # Spiral merge representing 2-adic convergence
        target_point = Dot(color=WHITE)
        self.place_in_area(target_point, "E4", "F6", scale_factor=1.0)
        target_label = MathTex("-1", color=WHITE, font_size=28).next_to(target_point, DOWN, buff=0.2)
        
        num_dots = 12
        dots = VGroup()
        for i in range(num_dots):
            dist = 1.2 * (0.7 ** i)
            angle = i * PI / 3
            dot_pos = target_point.get_center() + dist * np.array([np.cos(angle), np.sin(angle), 0])
            dot = Dot(dot_pos, radius=0.04, color=merge_color)
            dots.add(dot)
            
        self.play(FadeIn(target_point), FadeIn(target_label))
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.1))
        self.wait(0.5)
        
        # Final merge animation
        self.play(
            *[d.animate.move_to(target_point.get_center()).set_opacity(0) for d in dots],
            run_time=2
        )
        self.play(target_point.animate.scale(1.5).set_color(merge_color))
        self.wait(2)
