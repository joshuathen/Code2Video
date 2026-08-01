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
        # Initialize Scene
        lecture_lines = [
            "Can we find Pi using only simple fractions?",
            "A growing chain of ratios reveals Pi halves.",
            "This infinite product converges to one point five seven."
        ]
        self.setup_layout("The Hook: Finding Pi without a Compass", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Color change for current line
        self.lecture[0].set_color("#F4D03F")
        
        # Fade in a bright blue circle (#58C4DD) with 'Area = pi' (#FFFFFF) inside.
        # Radius adjusted to 0.45 to fit comfortably in the 2x1 grid area E4-F6
        circle = Circle(radius=0.45, color="#58C4DD", fill_opacity=0.6)
        area_label = Text("Area = π", color="#FFFFFF", font_size=20)
        
        # Fix 34: Positioned at bottom right to avoid equation area
        self.place_in_area(circle, "E4", "F6")
        self.place_in_area(area_label, "E4", "F6")
        
        self.play(FadeIn(circle), FadeIn(area_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Shift highlighting
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#F4D03F")
        
        # Pi half and equals sign
        pi_half = Text("π / 2", color="#FFFFFF", font_size=36)
        self.place_at_grid(pi_half, "B1", scale_factor=1.2)
        
        eq_sign = Text("=", color="#FFFFFF", font_size=36)
        self.place_at_grid(eq_sign, "B2", scale_factor=1.2)

        # ratios '2/1' and '2/3' appear (#F4D03F).
        r1 = Text("2/1", color="#F4D03F", font_size=32)
        dot1 = Text("·", color="#FFFFFF", font_size=36)
        r2 = Text("2/3", color="#F4D03F", font_size=32)
        
        # Grouping for consistent horizontal alignment
        ratio_grp_1 = VGroup(r1, dot1, r2).arrange(RIGHT, buff=0.15)
        # Fix 35: Placed at B3 for connectivity with equals sign at B2
        self.place_at_grid(ratio_grp_1, "B3", scale_factor=1.0)
        
        self.play(
            ReplacementTransform(circle, pi_half),
            FadeOut(area_label),
            FadeIn(eq_sign),
            FadeIn(ratio_grp_1)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Shift highlighting
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#F4D03F")
        
        # A sequence of ratios grows horizontally
        dot2 = Text("·", color="#FFFFFF", font_size=36)
        r3 = Text("4/3", color="#F4D03F", font_size=32)
        dot3 = Text("·", color="#FFFFFF", font_size=36)
        r4 = Text("4/5", color="#F4D03F", font_size=32)
        dots = Text("...", color="#FFFFFF", font_size=32)
        
        ratio_grp_2 = VGroup(dot2, r3, dot3, r4, dots).arrange(RIGHT, buff=0.15)
        # Fix 36: Positioned at C3-C5 for vertical alignment with ratio_grp_1
        self.place_in_area(ratio_grp_2, "C3", "C5", scale_factor=1.0)
        
        # Reaching the value 1.5707 (#FFFFFF)
        approx = Text("≈", color="#FFFFFF", font_size=36)
        self.place_at_grid(approx, "D1", scale_factor=1.2)
        
        # DecimalNumber uses Text-based logic to avoid latex dependency
        value_display = DecimalNumber(
            0,
            num_decimal_places=4,
            include_sign=False,
            color="#FFFFFF",
            mob_class=Text
        )
        self.place_at_grid(value_display, "D2", scale_factor=1.2)
        
        self.play(FadeIn(ratio_grp_2))
        self.play(FadeIn(approx), FadeIn(value_display))
        self.play(
            value_display.animate.set_value(1.5707),
            run_time=2,
            rate_func=bezier([0, 0, 1, 1])
        )
        self.wait(3)
