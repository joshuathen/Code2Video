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
        # Title and Lecture lines
        lecture_lines = [
            "Suddenly, the sensor beeps!",
            "We discard regions where the sensor remained silent.",
            "Our universe shrinks to only the triggered areas."
        ]
        self.setup_layout("The Geometric Shift: The New Universe", lecture_lines)

        # Colors
        COLOR_TP = "#77DD77" # Green: Cat exists and sensor beeps
        COLOR_FP = "#FF6961" # Red: No cat and sensor beeps
        COLOR_FN = "#779ECB" # Blue: Cat exists and sensor silent
        COLOR_TN = "#CFCFCF" # Gray: No cat and sensor silent
        HIGHLIGHT_COLOR = YELLOW

        # === Pre-setup of the state (Representing the full square from section 3) ===
        # Widths: Cat col = 1.0, No-Cat col = 2.0.
        # Heights: Beep row (top) = 2.1, No-Beep row (bottom) = 0.9 (for Cat); 
        #          Beep row (top) = 0.6, No-Beep row (bottom) = 2.4 (for No-Cat)
        
        rect_tp = Rectangle(width=1.0, height=2.1, fill_opacity=0.8, fill_color=COLOR_TP, stroke_width=2)
        rect_fn = Rectangle(width=1.0, height=0.9, fill_opacity=0.8, fill_color=COLOR_FN, stroke_width=2)
        rect_fp = Rectangle(width=2.0, height=0.6, fill_opacity=0.8, fill_color=COLOR_FP, stroke_width=2)
        rect_tn = Rectangle(width=2.0, height=2.4, fill_opacity=0.8, fill_color=COLOR_TN, stroke_width=2)

        # Assemble the square logically
        col_cat = VGroup(rect_tp, rect_fn).arrange(DOWN, buff=0)
        col_no_cat = VGroup(rect_fp, rect_tn).arrange(DOWN, buff=0)
        square_vgroup = VGroup(col_cat, col_no_cat).arrange(RIGHT, buff=0, aligned_edge=UP)
        
        # Position the initial state (Issue 37 Fix)
        self.place_in_area(square_vgroup, "A2", "D5", scale_factor=0.9)
        
        # Add internal labels
        label_tp = Text("TP", font_size=16).move_to(rect_tp.get_center())
        label_fp = Text("FP", font_size=16).move_to(rect_fp.get_center())
        label_fn = Text("FN", font_size=16).move_to(rect_fn.get_center())
        label_tn = Text("TN", font_size=16).move_to(rect_tn.get_center())
        labels = VGroup(label_tp, label_fp, label_fn, label_tn)

        self.add(square_vgroup, labels)
        self.wait(1)

        # === Animation for Lecture Line 1 ===
        # "Suddenly, the sensor beeps!"
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Visual cue for the beep (expanding ripple)
        beep_wave = Circle(radius=0.1, color=HIGHLIGHT_COLOR, stroke_width=8).move_to(square_vgroup.get_center())
        self.play(
            beep_wave.animate.scale(40).set_stroke(opacity=0),
            run_time=1.2,
            rate_func=rush_from
        )
        self.remove(beep_wave)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # "We discard regions where the sensor remained silent."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # Fade out regions where sensor silent (FN and TN)
        self.play(
            FadeOut(rect_fn, shift=DOWN*0.5),
            FadeOut(rect_tn, shift=DOWN*0.5),
            FadeOut(label_fn),
            FadeOut(label_tn),
            rect_tp.animate.set_stroke(opacity=0.4),
            rect_fp.animate.set_stroke(opacity=0.4),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Our universe shrinks to only the triggered areas."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )

        # Final Universe dimensions: combined vertical rectangle.
        # Total height of original square is approx 3 units.
        # TP and FP are rearranged to fill this height.
        final_width = 1.6
        target_tp = Rectangle(width=final_width, height=2.0, fill_opacity=0.9, fill_color=COLOR_TP, stroke_width=3)
        target_fp = Rectangle(width=final_width, height=1.0, fill_opacity=0.9, fill_color=COLOR_FP, stroke_width=3)
        target_universe = VGroup(target_tp, target_fp).arrange(DOWN, buff=0)
        
        # Position at center of the right side grid (Issue 36 Fix)
        self.place_in_area(target_universe, "A2", "D5", scale_factor=0.8)
        
        # Final Label for the entire structure (Issue 35 Fix)
        universe_label = Text("The Universe of Evidence", font_size=24, color=WHITE)
        self.place_in_area(universe_label, "F2", "F5", scale_factor=0.7)
        
        # Clearer internal labels
        text_tp = Text("Evidence (True Positive)", font_size=18).move_to(target_tp.get_center())
        text_fp = Text("Evidence (False Positive)", font_size=18).move_to(target_fp.get_center())

        # Smooth scaling and movement to form the new rectangle
        self.play(
            Transform(rect_tp, target_tp),
            Transform(rect_fp, target_fp),
            Transform(label_tp, text_tp),
            Transform(label_fp, text_fp),
            FadeIn(universe_label),
            run_time=2.5
        )
        self.wait(2)

        # Cleanup: Reset line highlight
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
