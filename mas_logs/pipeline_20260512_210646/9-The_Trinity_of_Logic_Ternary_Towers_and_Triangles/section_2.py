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

class Section2Scene(TeachingScene):
    def construct(self):
        # Setup title and lecture lines as per Stage-3 instructions
        title_str = "The Rule Change: Constrained Towers of Hanoi"
        lines_str = [
            'Standard Hanoi allows moves between any two pegs.',
            'Restricted rules only permit moves between adjacent pegs.',
            'To reach peg two, disks must pass through one.'
        ]
        self.setup_layout(title_str, lines_str)
        
        # Colors
        PEG_COLOR = "#FFFFFF"
        DISK_COLOR = "#FF8C00"
        X_COLOR = "#FF0000"
        ARROW_COLOR = "#00FF00"
        HIGHLIGHT_COLOR = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        
        # Create Pegs (Rows B to E)
        peg0 = Line(self.grid["B2"], self.grid["E2"], color=PEG_COLOR, stroke_width=6)
        peg1 = Line(self.grid["B4"], self.grid["E4"], color=PEG_COLOR, stroke_width=6)
        peg2 = Line(self.grid["B6"], self.grid["E6"], color=PEG_COLOR, stroke_width=6)
        
        # Numeric labels at Row F (Issue 30)
        label0 = Text("0", color=PEG_COLOR, font_size=24)
        label1 = Text("1", color=PEG_COLOR, font_size=24)
        label2 = Text("2", color=PEG_COLOR, font_size=24)
        self.place_at_grid(label0, "F2", scale_factor=0.8)
        self.place_at_grid(label1, "F4", scale_factor=0.8)
        self.place_at_grid(label2, "F6", scale_factor=0.8)
        
        # Asset Disk (Issue 27) scaled and placed at E2 (Issue 29)
        disk = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/disk.svg")
        disk.set_color(DISK_COLOR)
        self.place_at_grid(disk, "E2", scale_factor=0.7)
        
        self.play(
            Create(peg0), Create(peg1), Create(peg2),
            FadeIn(label0), FadeIn(label1), FadeIn(label2),
            FadeIn(disk)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Draw forbidden path (staying below Row A to avoid title)
        forbidden_arc = ArcBetweenPoints(
            self.grid["C2"], self.grid["C6"], 
            angle=-TAU/4, color=X_COLOR, stroke_opacity=0.5
        )
        
        # Red X restriction symbol at B4 (Issue 31)
        x_size = 0.2
        x_line1 = Line(self.grid["B4"] + x_size*(UP+LEFT), self.grid["B4"] + x_size*(DOWN+RIGHT), color=X_COLOR, stroke_width=8)
        x_line2 = Line(self.grid["B4"] + x_size*(UP+RIGHT), self.grid["B4"] + x_size*(DOWN+LEFT), color=X_COLOR, stroke_width=8)
        forbidden_x = VGroup(x_line1, x_line2)
        
        self.play(Create(forbidden_arc), Create(forbidden_x))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        # Pathing constrained to peak at Row B (Issue 29)
        # Move disk 0 -> 1
        arrow1 = Arrow(self.grid["C2"], self.grid["C4"], color=ARROW_COLOR, buff=0.1)
        self.play(disk.animate.move_to(self.grid["B2"]), run_time=0.5)
        self.play(
            disk.animate.move_to(self.grid["B4"]),
            GrowArrow(arrow1),
            run_time=0.7
        )
        self.play(disk.animate.move_to(self.grid["E4"]), run_time=0.5)
        self.wait(0.5)
        
        # Move disk 1 -> 2
        arrow2 = Arrow(self.grid["C4"], self.grid["C6"], color=ARROW_COLOR, buff=0.1)
        self.play(disk.animate.move_to(self.grid["B4"]), run_time=0.5)
        self.play(
            disk.animate.move_to(self.grid["B6"]),
            GrowArrow(arrow2),
            run_time=0.7
        )
        self.play(disk.animate.move_to(self.grid["E6"]), run_time=0.5)
        self.wait(2)
