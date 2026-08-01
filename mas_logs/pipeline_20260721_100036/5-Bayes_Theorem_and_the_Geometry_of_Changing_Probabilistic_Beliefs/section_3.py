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
        # Title and Lecture Lines
        title_text = "The Evidence Filter (Likelihoods)"
        lecture_lines = [
            "Likelihoods describe how evidence appears under each scenario.",
            "If the Phoenix is present, finding a feather is likely.",
            "We shade a tall region within the Phoenix strip.",
            "If absent, finding a feather is just a fluke.",
            "We shade a tiny region in the non-Phoenix strip."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_H = "#ADD8E6"      # Light Blue
        COLOR_NOT_H = "#FFFF00"  # Yellow
        
        # === Animation for Lecture Line 1 ===
        # "Likelihoods describe how evidence appears under each scenario."
        
        # Strip H: Represents prior probability of Phoenix existence
        strip_h = Rectangle(width=1.0, height=4.0, stroke_color=WHITE, stroke_width=2)
        self.place_in_area(strip_h, 'B2', 'E2')
        
        # Strip not H: Represents prior probability of Phoenix non-existence
        strip_not_h = Rectangle(width=3.0, height=4.0, stroke_color=WHITE, stroke_width=2)
        self.place_in_area(strip_not_h, 'B3', 'E5')
        
        label_h = Text("Phoenix (H)", font_size=16, color=WHITE)
        self.place_at_grid(label_h, 'A2', scale_factor=0.8)
        
        label_not_h = Text("No Phoenix (not H)", font_size=16, color=WHITE)
        # Fix Issue 26: label 'No Phoenix (not H)' cut off.
        self.place_in_area(label_not_h, 'A3', 'A5', scale_factor=0.8)
        
        self.play(
            Create(strip_h),
            Create(strip_not_h),
            Write(label_h),
            Write(label_not_h),
            self.lecture[0].animate.set_color(WHITE)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "If the Phoenix is present, finding a feather is likely."
        self.play(
            self.lecture[1].animate.set_color(COLOR_H),
            strip_h.animate.set_stroke(COLOR_H, width=4)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "We shade a tall region within the Phoenix strip."
        # Likelihood P(E|H) = 0.9 -> Height 3.6 of 4.0
        fill_h = Rectangle(
            width=1.0, 
            height=3.6, 
            fill_color=COLOR_H, 
            fill_opacity=0.6, 
            stroke_width=0
        )
        # Position relative to strip_h
        fill_h.move_to(strip_h.get_bottom(), aligned_edge=DOWN)
        
        formula_h = MathTex("P(E|H) = 0.9", font_size=24, color=COLOR_H)
        # Fix Issue 28: formula 'P(E | H) = 0.9' overlaps with 'No Phoenix' strip area.
        self.place_at_grid(formula_h, 'D2', scale_factor=0.5)
        
        self.play(
            self.lecture[2].animate.set_color(COLOR_H),
            FadeIn(fill_h, shift=UP),
            Write(formula_h)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "If absent, finding a feather is just a fluke."
        self.play(
            self.lecture[3].animate.set_color(COLOR_NOT_H),
            strip_not_h.animate.set_stroke(COLOR_NOT_H, width=4)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "We shade a tiny region in the non-Phoenix strip."
        # Likelihood P(E|not H) = 0.05 -> Height 0.2 of 4.0
        fill_not_h = Rectangle(
            width=3.0, 
            height=0.2, 
            fill_color=COLOR_NOT_H, 
            fill_opacity=0.6, 
            stroke_width=0
        )
        # Position relative to strip_not_h
        fill_not_h.move_to(strip_not_h.get_bottom(), aligned_edge=DOWN)
        
        formula_not_h = MathTex("P(E|\\text{not } H) = 0.05", font_size=24, color=COLOR_NOT_H)
        # Fix Issue 27: formula 'P(E | not H) = 0.05' off-screen at bottom.
        self.place_at_grid(formula_not_h, 'E5', scale_factor=0.6)
        
        self.play(
            self.lecture[4].animate.set_color(COLOR_NOT_H),
            FadeIn(fill_not_h, shift=UP),
            Write(formula_not_h)
        )
        self.wait(2)
