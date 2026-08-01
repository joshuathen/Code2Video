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
        # Configuration
        title_text = "Defining the 2-adic Metric"
        lecture_lines = [
            "We define the 2-adic absolute value using this valuation.",
            "More factors of two make a number's value smaller.",
            "A scale shows 16 weighing less than 2.",
            "Large powers of two sit extremely close to zero.",
            "This metric flips our traditional sense of size."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        blue_color = "#ADD8E6"
        # Asset path from storyboard
        scale_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg"

        # === Animation for Lecture Line 1 ===
        # Write |x|2 = 2^-v2(x) in light blue #ADD8E6.
        # Fix Issue 24: Correct positioning and scale
        formula = MathTex(r"|x|_2 = 2^{-v_2(x)}", color=blue_color)
        self.place_in_area(formula, 'B2', 'B5', scale_factor=1.2)
        
        self.play(self.lecture[0].animate.set_color(blue_color))
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # More factors of two make a number's value smaller.
        self.play(self.lecture[1].animate.set_color(blue_color))
        # Highlight formula to emphasize relationship
        self.play(Indicate(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # A scale shows 16 weighing less than 2.
        self.play(self.lecture[2].animate.set_color(blue_color))
        
        # Scale Setup
        # Fix Issue 26: Use SVG asset as fulcrum/base and place in area F3-F4
        # [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg]
        fulcrum = SVGMobject(scale_path, color=WHITE)
        self.place_in_area(fulcrum, 'F3', 'F4', scale_factor=0.8)
        pivot_point = fulcrum.get_top()

        angle_tracker = ValueTracker(0)

        # Beam and Pans functions
        # side -1 for left, 1 for right
        def get_pos(side, offset_y=0):
            a = angle_tracker.get_value()
            # Rotation vector relative to pivot
            vec = np.array([side * 1.5 * np.cos(a), side * 1.5 * np.sin(a), 0])
            return pivot_point + vec + UP * offset_y

        beam = Line(color=WHITE)
        beam.add_updater(lambda m: m.put_start_and_end_on(get_pos(-1), get_pos(1)))
        
        pan_l = Line(color=WHITE)
        pan_l.add_updater(lambda m: m.put_start_and_end_on(get_pos(-1), get_pos(-1) + DOWN * 0.4))
        
        pan_r = Line(color=WHITE)
        pan_r.add_updater(lambda m: m.put_start_and_end_on(get_pos(1), get_pos(1) + DOWN * 0.4))

        # Weights
        # |2|_2 = 0.5 (Visually larger weight)
        w2_box = Square(side_length=0.7, color=WHITE, fill_opacity=0.4)
        w2_label = Text("2", font_size=24)
        w2 = VGroup(w2_box, w2_label)
        w2.add_updater(lambda m: m.move_to(get_pos(-1) + UP * 0.35))

        # |16|_2 = 0.0625 (Visually tiny weight)
        # Shrink the weights visually as they are added (Issue 17)
        w16_box = Square(side_length=0.2, color=WHITE, fill_opacity=0.4)
        w16_label = Text("16", font_size=12)
        w16 = VGroup(w16_box, w16_label)
        w16.add_updater(lambda m: m.move_to(get_pos(1) + UP * 0.1))

        self.add(fulcrum, beam, pan_l, pan_r)
        self.play(FadeIn(w2), FadeIn(w16))
        
        # 16 is "lighter" 2-adically, so 2 (left) goes down.
        # Positive angle means right side goes up, left side goes down.
        self.play(angle_tracker.animate.set_value(20 * DEGREES), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Large powers of two sit extremely close to zero.
        self.play(self.lecture[3].animate.set_color(blue_color))
        
        # Adding 4 and 8 to the lighter side (right)
        # Shrink the weights visually (4 is smaller than 2, 8 is smaller than 4)
        w4 = VGroup(Square(side_length=0.4, color=WHITE, fill_opacity=0.4), Text("4", font_size=18))
        w4.add_updater(lambda m: m.move_to(get_pos(1) + UP * 0.3 + LEFT * 0.3))
        
        w8 = VGroup(Square(side_length=0.3, color=WHITE, fill_opacity=0.4), Text("8", font_size=15))
        w8.add_updater(lambda m: m.move_to(get_pos(1) + UP * 0.25 + RIGHT * 0.3))
        
        self.play(FadeIn(w4), FadeIn(w8))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # This metric flips our traditional sense of size.
        self.play(self.lecture[4].animate.set_color(blue_color))
        
        # Fix Issue 25: Reposition comparison to C3-D4
        comparison = MathTex(r"|16|_2 < |2|_2", color=blue_color)
        self.place_in_area(comparison, 'C3', 'D4', scale_factor=1.0)
        self.play(Write(comparison))
        
        self.wait(3)
