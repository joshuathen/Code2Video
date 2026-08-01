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
        lecture_lines = [
            "Let's record the region count for each point added.",
            "The total regions consistently double at every step.",
            "Does this pattern hold for the sixth point?"
        ]
        self.setup_layout("The Alluring Pattern (n = 1 to 5)", lecture_lines)
        
        # Define Colors
        c1 = BLUE_A
        c2 = GREEN
        c3 = YELLOW
        
        # Define Table Elements
        # Column 2: Points, Column 4: Regions (Issue 47: Moved from 5 to 4)
        header_pts = Text("Points", font_size=24, color=WHITE)
        header_regs = Text("Regions", font_size=24, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(c1))
        
        # Row A: Header
        self.place_at_grid(header_pts, "A2", scale_factor=0.7)
        self.place_at_grid(header_regs, "A4", scale_factor=0.7)
        
        # Row B: n=1 (1 region)
        row1_n = Text("1", font_size=24, color=c1)
        row1_r = Text("1", font_size=24, color=c1)
        self.place_at_grid(row1_n, "B2", scale_factor=0.7)
        self.place_at_grid(row1_r, "B4", scale_factor=0.7)
        
        self.play(FadeIn(header_pts), FadeIn(header_regs), Write(row1_n), Write(row1_r))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(c2))
        
        rows_n = []
        rows_r = []
        arrows = VGroup()
        x2_labels = VGroup()
        
        # Define n=2 to n=5 rows
        # Issue 47: Used rows C to F to keep them on screen. Scale 0.7.
        data = [
            (2, 2, "B", "C"),
            (3, 4, "C", "D"),
            (4, 8, "D", "E"),
            (5, 16, "E", "F")
        ]
        
        for n_val, r_val, prev_row, curr_row in data:
            rn = Text(str(n_val), font_size=24, color=c2)
            rr = Text(str(r_val), font_size=24, color=c2)
            self.place_at_grid(rn, f"{curr_row}2", scale_factor=0.7)
            self.place_at_grid(rr, f"{curr_row}4", scale_factor=0.7)
            rows_n.append(rn)
            rows_r.append(rr)
            
            # Doubling Arrow from previous row to current row (in Regions col)
            # Position arrows between rows in column 4/5
            start_pos = self.grid[f"{prev_row}4"]
            end_pos = self.grid[f"{curr_row}4"]
            
            # Arrow offset to the right
            arr = CurvedArrow(start_pos + RIGHT*0.35, end_pos + RIGHT*0.35, angle=-PI/4, color=GREEN, stroke_width=2).scale(0.6)
            # Label "x2" in column 5
            lbl = Text("x2", font_size=16, color=GREEN)
            lbl.move_to(self.grid[f"{curr_row}5"] + LEFT*0.3)
            
            arrows.add(arr)
            x2_labels.add(lbl)
            
        self.play(
            AnimationGroup(
                *[Write(obj) for obj in rows_n + rows_r],
                Create(arrows),
                Write(x2_labels),
                lag_ratio=0.3
            )
        )
        self.wait(1)
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(c3))
        
        # Pulse Regions Column
        regions_vgroup = VGroup(row1_r, *rows_r)
        self.play(regions_vgroup.animate.set_color(YELLOW))
        
        # Show Prediction for n=6
        # Issue 47: Row G (calculated as 1 unit below F)
        pos_n6 = self.grid["F2"] + DOWN * 1.0
        pos_r6 = self.grid["F4"] + DOWN * 1.0
        
        pred_label = Text("6", font_size=24, color=c3).move_to(pos_n6)
        
        # Issue 47: Prediction "32?" scaled down
        prediction = Text("32?", font_size=28, color=YELLOW).move_to(pos_r6)
        
        # Final arrow for n=6
        final_arrow = CurvedArrow(self.grid["F4"] + RIGHT*0.35, pos_r6 + RIGHT*0.35, angle=-PI/4, color=YELLOW, stroke_width=3).scale(0.6)
        final_x2 = Text("x2", font_size=18, color=YELLOW).move_to(self.grid["F5"] + DOWN*1.0 + LEFT*0.3)
        
        # Rule Text (top right of table area)
        rule_text = Text("Rule: 2^(n-1)?", font_size=24, color=WHITE)
        self.place_at_grid(rule_text, "B5", scale_factor=0.9)
        
        self.play(Write(pred_label), Write(prediction), Create(final_arrow), Write(final_x2), FadeIn(rule_text))
        
        # Pulsing Prediction
        for _ in range(3):
            self.play(prediction.animate.scale(1.15), run_time=0.3)
            self.play(prediction.animate.scale(1/1.15), run_time=0.3)
            
        self.wait(2)
