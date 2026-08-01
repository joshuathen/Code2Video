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
        self.setup_layout(
            "Introduction: The Restricted Puzzle", 
            [
                'Here is the classic Towers of Hanoi puzzle setup.', 
                'Direct jumps between Peg A and C are forbidden.', 
                'Disks can only move between adjacent neighboring pegs.', 
                'Every shift must pass through the middle peg, B.', 
                'Our goal is moving all disks from A to C.'
            ]
        )
        
        # Colors
        COLOR_PEG = "#FFFFFF"
        COLOR_DISK1 = "#FF5733" # Large
        COLOR_DISK2 = "#33FF57" # Medium
        COLOR_DISK3 = "#3357FF" # Small
        COLOR_ARROW = "#00FF00"
        COLOR_FORBIDDEN = "#FF0000"
        HIGHLIGHT_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Create Pegs
        peg_a = Line(self.grid["B2"], self.grid["E2"], color=COLOR_PEG, stroke_width=8)
        peg_b = Line(self.grid["B4"], self.grid["E4"], color=COLOR_PEG, stroke_width=8)
        peg_c = Line(self.grid["B6"], self.grid["E6"], color=COLOR_PEG, stroke_width=8)
        
        label_a = Text("A", font_size=24, color=COLOR_PEG)
        label_b = Text("B", font_size=24, color=COLOR_PEG)
        label_c = Text("C", font_size=24, color=COLOR_PEG)
        self.place_at_grid(label_a, "F2")
        self.place_at_grid(label_b, "F4")
        self.place_at_grid(label_c, "F6")

        # Create Disks (Initial state on Peg A)
        # disk1 is large/bottom, disk3 is small/top
        disk1 = RoundedRectangle(corner_radius=0.1, width=1.4, height=0.4, fill_opacity=1, fill_color=COLOR_DISK1, stroke_width=0)
        disk2 = RoundedRectangle(corner_radius=0.1, width=1.0, height=0.4, fill_opacity=1, fill_color=COLOR_DISK2, stroke_width=0)
        disk3 = RoundedRectangle(corner_radius=0.1, width=0.6, height=0.4, fill_opacity=1, fill_color=COLOR_DISK3, stroke_width=0)
        
        self.place_at_grid(disk1, "E2") # Corrected by Issue 28
        self.place_at_grid(disk2, "D2")
        self.place_at_grid(disk3, "C2") # Corrected by Issue 27

        self.play(
            FadeIn(peg_a), FadeIn(peg_b), FadeIn(peg_c),
            FadeIn(label_a), FadeIn(label_b), FadeIn(label_c),
            FadeIn(disk1), FadeIn(disk2), FadeIn(disk3)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Red cross over a dashed curved arrow between Peg A and Peg C
        jump_arrow_base = CurvedArrow(self.grid["B2"] + UP*0.4, self.grid["B6"] + UP*0.4, angle=-PI/2, color=COLOR_FORBIDDEN)
        jump_arrow = DashedVMobject(jump_arrow_base, num_dashes=20)
        
        cross_line1 = Line(UP+LEFT, DOWN+RIGHT, color=COLOR_FORBIDDEN, stroke_width=10).scale(0.5)
        cross_line2 = Line(UP+RIGHT, DOWN+LEFT, color=COLOR_FORBIDDEN, stroke_width=10).scale(0.5)
        red_cross = VGroup(cross_line1, cross_line2)
        self.place_in_area(red_cross, 'B3', 'B5', scale_factor=0.8) # Corrected by Issue 29
        
        self.play(Create(jump_arrow), FadeIn(red_cross))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Double arrows between pegs to indicate adjacency restriction
        arrow_ab = DoubleArrow(self.grid["E2"] + RIGHT*0.7, self.grid["E4"] + LEFT*0.7, color=COLOR_ARROW, stroke_width=4, tip_length=0.2)
        arrow_bc = DoubleArrow(self.grid["E4"] + RIGHT*0.7, self.grid["E6"] + LEFT*0.7, color=COLOR_ARROW, stroke_width=4, tip_length=0.2)
        
        self.play(Create(arrow_ab), Create(arrow_bc))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Animate the smallest disk (disk3) sliding from Peg A to Peg B, then from Peg B to Peg C
        self.play(disk3.animate.move_to(self.grid["C4"]), run_time=1)
        self.play(disk3.animate.move_to(self.grid["C6"]), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        # Snap other disks to C to show target state
        self.play(
            disk1.animate.move_to(self.grid["E6"]),
            disk2.animate.move_to(self.grid["D6"]),
            run_time=1.5
        )
        
        # Flash the stack on C
        self.play(Flash(self.grid["D6"], color=HIGHLIGHT_COLOR, flash_radius=1.2))
        
        # Highlight start (A) and end (C) pegs labels
        rect_a = SurroundingRectangle(label_a, color=HIGHLIGHT_COLOR, buff=0.1)
        rect_c = SurroundingRectangle(label_c, color=HIGHLIGHT_COLOR, buff=0.1)
        self.play(Create(rect_a), Create(rect_c))
        
        self.wait(2)
