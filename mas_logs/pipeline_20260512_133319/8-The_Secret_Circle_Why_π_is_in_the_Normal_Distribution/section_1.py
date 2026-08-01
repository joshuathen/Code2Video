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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initial Setup
        title = "The Mystery of the Bell Curve"
        lines = [
            'The Normal Distribution describes many natural phenomena.',
            'Notice the constant pi hidden in this linear formula.',
            'Imagine an archer shooting at a circular target.',
            'His errors stack up to form this bell shape.',
            'Why does a linear curve require circular geometry?'
        ]
        self.setup_layout(title, lines)

        # Pre-define Mobjects
        # 1. Formula
        f_part = Text("f(x) =", font_size=24, color=WHITE)
        num = Text("1", font_size=24, color=WHITE)
        frac_line = Line(LEFT*0.5, RIGHT*0.5, stroke_width=2, color=WHITE)
        den_sigma = Text("σ", font_size=24, color=WHITE)
        den_sqrt = Text("√", font_size=24, color=WHITE)
        den_two = Text("2", font_size=24, color=WHITE)
        den_pi = Text("π", font_size=24, color=WHITE)
        
        den_group = VGroup(den_sigma, den_sqrt, den_two, den_pi).arrange(RIGHT, buff=0.05)
        num.next_to(frac_line, UP, buff=0.1)
        den_group.next_to(frac_line, DOWN, buff=0.1)
        fraction = VGroup(num, frac_line, den_group)
        
        e_part = Text("e", font_size=24, color=WHITE)
        exponent = Text("-½((x-μ)/σ)²", font_size=16, color=WHITE)
        exponent.next_to(e_part.get_corner(UR), RIGHT, buff=0.05).shift(UP*0.1)
        
        formula = VGroup(f_part, fraction, e_part, exponent).arrange(RIGHT, buff=0.2)
        # Fix Issue 28: Scale factor 0.7
        self.place_in_area(formula, "A2", "A5", scale_factor=0.7)

        # 2. Bell Curve
        curve = VMobject(color="#58C4DD")
        points = []
        for x_val in np.arange(-2.5, 2.51, 0.1):
            y_val = 2.0 * np.exp(- (x_val**2) / (2 * 0.8**2))
            points.append(np.array([x_val, y_val, 0]))
        curve.set_points_smoothly(points)
        # Fix Issue 29: Scale factor 0.7
        self.place_in_area(curve, "B2", "D5", scale_factor=0.7)

        # 3. Target and Archer Logic
        target_center = self.grid["E3"]
        target_rings = VGroup(*[
            Circle(radius=r, color=WHITE, stroke_width=1) 
            for r in [0.3, 0.6, 0.9]
        ]).move_to(target_center)
        
        np.random.seed(42)
        hit_coords = []
        for _ in range(20):
            angle = np.random.uniform(0, 2 * PI)
            radius = abs(np.random.normal(0, 0.45))
            x_hit = radius * np.cos(angle)
            y_hit = radius * np.sin(angle)
            hit_coords.append((x_hit, y_hit))

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#58C4DD")
        self.play(Create(curve), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Write(formula), run_time=1.5)
        self.play(
            den_pi.animate.set_color(YELLOW),
            Indicate(den_pi, color=YELLOW, scale_factor=1.4),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(WHITE)
        self.play(FadeIn(target_rings))
        
        first_dots = VGroup()
        for i in range(8):
            x_h, y_h = hit_coords[i]
            dot = Dot(point=target_center + np.array([x_h, y_h, 0]), radius=0.04, color=WHITE)
            first_dots.add(dot)
            self.play(FadeIn(dot, scale=0.5), run_time=0.15)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#58C4DD")
        
        histogram_base_y = self.grid["F3"][1] - 0.5
        bin_width = 0.2
        bins = {} 
        bars = VGroup()

        # Animate projection of existing dots and adding new ones
        all_dots = list(first_dots)
        for i in range(8, 20):
            x_h, y_h = hit_coords[i]
            all_dots.append(Dot(point=target_center + np.array([x_h, y_h, 0]), radius=0.04, color=WHITE))

        for i, dot in enumerate(all_dots):
            if i >= 8: self.add(dot)
            
            x_h, y_h = hit_coords[i]
            target_pos = dot.get_center()
            proj_x = target_pos[0]
            
            bin_idx = round(x_h / bin_width)
            bins[bin_idx] = bins.get(bin_idx, 0) + 1
            
            bar_height = bins[bin_idx] * 0.12
            bar = Rectangle(
                width=bin_width * 0.7, 
                height=bar_height, 
                fill_opacity=0.8, 
                fill_color="#58C4DD", 
                stroke_width=0
            )
            bar.move_to(np.array([proj_x, histogram_base_y + bar_height/2, 0]))
            
            self.play(dot.animate.move_to(np.array([proj_x, histogram_base_y, 0])), run_time=0.1)
            self.add(bar)
            bars.add(bar)

        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # Fix Issue 27: Position D6, scale 0.8
        question_mark = Text("?", font_size=60, color=YELLOW)
        self.place_at_grid(question_mark, "D6", scale_factor=0.8)
        
        glow_pi = den_pi.copy().set_color(YELLOW).scale(1.2).set_opacity(0.5)
        glow_target = target_rings.copy().set_color(YELLOW).scale(1.1).set_opacity(0.3)

        self.play(
            FadeIn(question_mark),
            FadeIn(glow_pi),
            FadeIn(glow_target),
            target_rings.animate.set_color(YELLOW),
            run_time=1.5
        )
        self.wait(2)
