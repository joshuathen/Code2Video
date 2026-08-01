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

class Section6Scene(TeachingScene):
    def construct(self):
        # Data
        title_text = "Application: The Squirrel's Cache"
        lecture_lines = [
            "A squirrel's nut collection doubles every single day.",
            "How many days to reach sixty-four nuts?",
            "The empty top vertex represents this logarithm problem."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_SQUIRREL = "#A52A2A" # Brown
        COLOR_BASE = "#A52A2A"    # Matches squirrel (2)
        COLOR_RESULT = "#ADD8E6"  # Light Blue (64)
        COLOR_EXP = "#FFFF00"     # Yellow (?)
        COLOR_TRIANGLE = "#FFFFFF" # White
        
        # Positions
        pos_bl = self.grid["E2"]
        pos_br = self.grid["E5"]
        # Midpoint of B3 and B4 for the top vertex
        pos_top = (self.grid["B3"] + self.grid["B4"]) / 2
        
        # Mobjects
        # Triangle
        line_base = Line(pos_bl, pos_br, color=COLOR_TRIANGLE)
        line_left = Line(pos_bl, pos_top, color=COLOR_TRIANGLE)
        line_right = Line(pos_br, pos_top, color=COLOR_TRIANGLE)
        triangle = VGroup(line_base, line_left, line_right)
        
        # Squirrel - represented by a brown circle and label
        squirrel = Circle(radius=0.25, color=COLOR_SQUIRREL, fill_opacity=1.0)
        squirrel_label = Text("Squirrel", font_size=16, color=WHITE).next_to(squirrel, DOWN, buff=0.1)
        squirrel_grp = VGroup(squirrel, squirrel_label)
        
        # Values
        val_2 = Text("2", font_size=36, color=COLOR_BASE)
        self.place_at_grid(val_2, "E2")
        
        val_64 = Text("64", font_size=36, color=COLOR_RESULT)
        self.place_at_grid(val_64, "E5")
        
        val_q = Text("?", font_size=42, color=COLOR_EXP)
        # Issue 35 fix: increased scale factor to 1.5 for better prominence
        self.place_in_area(val_q, "B3", "B4", scale_factor=1.5)
        
        # === Animation for Lecture Line 1 ===
        # A squirrel's nut collection doubles every single day.
        self.play(self.lecture[0].animate.set_color(COLOR_BASE))
        self.play(Create(triangle), run_time=1)
        
        # Issue 34 fix: start at F2 instead of F1 to avoid being too close to the lecture area
        self.place_at_grid(squirrel_grp, "F2")
        self.play(FadeIn(squirrel_grp))
        # Move squirrel near the Base vertex
        self.play(squirrel_grp.animate.move_to(pos_bl + DOWN * 0.7))
        self.play(Write(val_2))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # How many days to reach sixty-four nuts?
        self.play(self.lecture[1].animate.set_color(COLOR_RESULT))
        # Move squirrel near the Result vertex
        self.play(squirrel_grp.animate.move_to(pos_br + DOWN * 0.7))
        self.play(Write(val_64))
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        # The empty top vertex represents this logarithm problem.
        self.play(self.lecture[2].animate.set_color(COLOR_EXP))
        # Move squirrel away to highlight the top
        self.play(squirrel_grp.animate.move_to(self.grid["F3"]))
        self.play(Write(val_q))
        self.play(Indicate(val_q, color=COLOR_EXP))
        self.wait(2)
