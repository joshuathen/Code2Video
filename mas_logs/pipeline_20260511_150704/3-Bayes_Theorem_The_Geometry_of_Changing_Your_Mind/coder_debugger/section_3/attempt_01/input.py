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
        # Initial Setup
        title = "The Geometry of New Information"
        lines = [
            'When the camera glints, non-glinting outcomes become impossible.', 
            'We focus only on the regions where evidence occurred.', 
            'The rest of the sample space is now discarded.'
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_GOLD = "#FFD700"
        COLOR_GREY = "#888888"
        COLOR_DISCARD = "#222222"
        COLOR_OUTLINE = "#FFFFFF"

        # Define the geometric components
        # Gold Strip (Prior for Gold Coin) - Split into Glint (~80%) and No Glint (~20%)
        # A to E is 5 units, F is 1 unit. Total 6 units.
        gold_glint = Rectangle(width=3, height=5, fill_color=COLOR_GOLD, fill_opacity=0.8, stroke_width=0)
        gold_no_glint = Rectangle(width=3, height=1, fill_color=COLOR_GOLD, fill_opacity=0.8, stroke_width=0)
        
        # Grey Strip (Prior for Fair Coin) - Split into Glint (~20%) and No Glint (~80%)
        # A is 1 unit, B to F is 5 units. Total 6 units.
        grey_glint = Rectangle(width=3, height=1, fill_color=COLOR_GREY, fill_opacity=0.8, stroke_width=0)
        grey_no_glint = Rectangle(width=3, height=5, fill_color=COLOR_GREY, fill_opacity=0.8, stroke_width=0)

        # Labels simulating inheritance from previous section
        label_gold = Text("Gold", font_size=20, color=WHITE)
        label_grey = Text("Grey", font_size=20, color=WHITE)
        prob_label_1 = Text("0.8", font_size=18, color=WHITE)
        prob_label_2 = Text("0.2", font_size=18, color=WHITE)

        # Position elements
        self.place_in_area(gold_glint, "A1", "E3")
        self.place_in_area(gold_no_glint, "F1", "F3")
        self.place_in_area(grey_glint, "A4", "A6")
        self.place_in_area(grey_no_glint, "B4", "F6")

        # Fixed positions based on Issue 28 and 29
        self.place_at_grid(label_gold, "A2", scale_factor=0.8)
        self.place_at_grid(prob_label_1, "C2", scale_factor=0.8)
        self.place_at_grid(label_grey, "A5", scale_factor=0.6)
        self.place_at_grid(prob_label_2, "A6", scale_factor=0.6)

        # Initial appearance (re-establishing state)
        self.add(gold_glint, gold_no_glint, grey_glint, grey_no_glint)
        self.add(label_gold, label_grey, prob_label_1, prob_label_2)

        # === Animation for Lecture Line 1 ===
        # When the camera glints, non-glinting outcomes become impossible.
        # Bottom 20% of gold and bottom 80% of grey turn dark grey.
        self.play(
            self.lecture[0].animate.set_color(COLOR_GOLD),
            gold_no_glint.animate.set_fill(COLOR_DISCARD),
            grey_no_glint.animate.set_fill(COLOR_DISCARD),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # We focus only on the regions where evidence occurred.
        # Create bold white outlines around the surviving regions.
        outline_gold = Rectangle(width=3, height=5, stroke_color=COLOR_OUTLINE, stroke_width=6, fill_opacity=0)
        outline_grey = Rectangle(width=3, height=1, stroke_color=COLOR_OUTLINE, stroke_width=6, fill_opacity=0)
        self.place_in_area(outline_gold, "A1", "E3")
        self.place_in_area(outline_grey, "A4", "A6")

        self.play(
            self.lecture[1].animate.set_color(COLOR_OUTLINE),
            Create(outline_gold),
            Create(outline_grey),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The rest of the sample space is now discarded.
        # Fade out labels to clean up the visual space.
        self.play(
            self.lecture[2].animate.set_color(COLOR_GREY),
            FadeOut(label_gold),
            FadeOut(label_grey),
            FadeOut(prob_label_1),
            FadeOut(prob_label_2),
            run_time=2
        )
        self.wait(2)
