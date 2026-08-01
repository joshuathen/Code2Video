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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup the standard layout with the specific teaching content script
        lecture_lines = [
            "A subtle balance exists between these two prime lanes.",
            "Alternating fractions from each track create a harmonic sum.",
            "This series converges exactly to Pi over four."
        ]
        self.setup_layout("The Prime Race and the Bridge to Pi", lecture_lines)

        # Pre-define colors
        SILVER = "#C0C0C0"
        GREEN = "#00FF00"
        HIGHLIGHT = YELLOW

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT)
        
        # Create scale components
        pivot = Triangle(color=SILVER).scale(0.3)
        self.place_in_area(pivot, "F3", "F4")
        
        beam = Line(LEFT*2.5, RIGHT*2.5, color=SILVER, stroke_width=6)
        self.place_in_area(beam, "D2", "D5")
        
        # Pans
        left_pan = VGroup(
            Line(ORIGIN, DOWN*0.8, color=SILVER),
            Arc(radius=0.8, start_angle=PI, angle=PI, color=SILVER)
        ).shift(beam.get_left())
        
        right_pan = VGroup(
            Line(ORIGIN, DOWN*0.8, color=SILVER),
            Arc(radius=0.8, start_angle=PI, angle=PI, color=SILVER)
        ).shift(beam.get_right())
        
        # Leibniz formula above the scale
        formula = Text("π/4 = 1 - 1/3 + 1/5 - 1/7 + ...", color=WHITE, font_size=28)
        self.place_in_area(formula, "A2", "B5", scale_factor=0.9)
        
        self.play(
            Create(VGroup(pivot, beam, left_pan, right_pan)),
            Write(formula),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT)
        
        # Define fractional terms at specific non-overlapping grid positions (Resolves Issue 29 and 30)
        # Left lane (+): 1, 1/5, 1/9
        term_1 = Text("1", font_size=32)
        self.place_at_grid(term_1, "D1")
        
        term_1_5 = Text("1/5", font_size=32)
        self.place_at_grid(term_1_5, "E1")
        
        term_1_9 = Text("1/9", font_size=32)
        self.place_at_grid(term_1_9, "E2")
        
        # Right lane (-): 1/3, 1/7, 1/11
        term_1_3 = Text("1/3", font_size=32)
        self.place_at_grid(term_1_3, "D6")
        
        term_1_7 = Text("1/7", font_size=32)
        self.place_at_grid(term_1_7, "E6")
        
        term_1_11 = Text("1/11", font_size=32)
        self.place_at_grid(term_1_11, "E5")
        
        # Fade them in first
        all_terms = VGroup(term_1, term_1_3, term_1_5, term_1_7, term_1_9, term_1_11)
        self.play(FadeIn(all_terms))
        self.wait(0.5)
        
        # Drop to pans
        pan_l_pos = left_pan.get_center() + DOWN*0.2
        pan_r_pos = right_pan.get_center() + DOWN*0.2
        
        self.play(
            term_1.animate.move_to(pan_l_pos),
            term_1_5.animate.move_to(pan_l_pos + UP*0.2),
            term_1_9.animate.move_to(pan_l_pos + UP*0.4),
            term_1_3.animate.move_to(pan_r_pos),
            term_1_7.animate.move_to(pan_r_pos + UP*0.2),
            term_1_11.animate.move_to(pan_r_pos + UP*0.4),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT)
        
        # Final result label positioned clear of the pivot (Resolves Issue 31)
        result_label = Text("Result: π/4", color=GREEN, font_size=24)
        self.place_at_grid(result_label, "F5", scale_factor=0.8)
        
        # Scale wiggles to indicate finding balance
        self.play(
            beam.animate.rotate(0.05, about_point=pivot.get_top()),
            VGroup(left_pan, term_1, term_1_5, term_1_9).animate.shift(UP*0.1),
            VGroup(right_pan, term_1_3, term_1_7, term_1_11).animate.shift(DOWN*0.1),
            run_time=1.5, rate_func=wiggle
        )
        
        self.play(FadeIn(result_label))
        self.wait(3)
