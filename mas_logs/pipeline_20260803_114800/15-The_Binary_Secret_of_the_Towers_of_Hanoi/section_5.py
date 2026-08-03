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

class Section5Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Move one is binary zero-zero-one; move disk one.",
            "Move two is zero-one-zero; move disk two.",
            "Move three is zero-one-one; move disk one again.",
            "Move four is one-zero-zero; move disk three.",
            "Binary counting guides every single disk placement perfectly."
        ]
        self.setup_layout("Animated Walkthrough: 3-Disk Demo", lecture_lines)

        # Assets/Colors
        COLOR_DISK1 = "#FF6347"  # Tomato
        COLOR_DISK2 = "#32CD32"  # LimeGreen
        COLOR_DISK3 = "#1E90FF"  # DodgerBlue
        COLOR_PEGS = "#888888"
        HIGHLIGHT_COLOR = "#FFFF00"
        DISK_ASSET = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"

        # Pegs - Using visual grid
        peg1 = Line(self.grid["C2"], self.grid["F2"], color=COLOR_PEGS, stroke_width=8)
        peg2 = Line(self.grid["C4"], self.grid["F4"], color=COLOR_PEGS, stroke_width=8)
        peg3 = Line(self.grid["C6"], self.grid["F6"], color=COLOR_PEGS, stroke_width=8)
        
        base = Line(self.grid["F1"] + LEFT*0.5, self.grid["F6"] + RIGHT*0.5, color=COLOR_PEGS, stroke_width=4)
        
        # Labels for Pegs
        peg1_label = Text("Peg 1", font_size=18).next_to(peg1, DOWN, buff=0.2)
        peg2_label = Text("Peg 2", font_size=18).next_to(peg2, DOWN, buff=0.2)
        peg3_label = Text("Peg 3", font_size=18).next_to(peg3, DOWN, buff=0.2)

        # Disks - Using SVGMobject and adjusting scale for relative sizes
        disk1 = SVGMobject(DISK_ASSET).set_color(COLOR_DISK1).set_width(0.6)
        disk2 = SVGMobject(DISK_ASSET).set_color(COLOR_DISK2).set_width(1.0)
        disk3 = SVGMobject(DISK_ASSET).set_color(COLOR_DISK3).set_width(1.4)

        # Initial Placement (Peg 1) and scaling per VideoCritic (Issue 29)
        self.place_at_grid(disk3, "F2", scale_factor=0.8)
        self.place_at_grid(disk2, "E2", scale_factor=0.8)
        self.place_at_grid(disk1, "D2", scale_factor=0.8)

        # Binary Counter UI - Positioned at A5 per VideoCritic (Issue 28)
        counter_label = Text("Step:", font_size=24)
        self.place_at_grid(counter_label, "A5", scale_factor=1.0)
        
        # Initial binary value
        binary_val = Text("000", font_size=32, color=YELLOW)
        binary_val.next_to(counter_label, RIGHT, buff=0.3)

        self.add(peg1, peg2, peg3, base, peg1_label, peg2_label, peg3_label, disk3, disk2, disk1, counter_label, binary_val)

        def update_binary(val):
            new_text = Text(f"{val:03b}", font_size=32, color=YELLOW).move_to(binary_val)
            return Transform(binary_val, new_text)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR), run_time=0.5)
        
        self.play(
            update_binary(1),
            Succession(
                disk1.animate.move_to(self.grid["B2"]),
                disk1.animate.move_to(self.grid["B6"]),
                disk1.animate.move_to(self.grid["F6"])
            ),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR),
            run_time=0.5
        )
        
        self.play(
            update_binary(2),
            Succession(
                disk2.animate.move_to(self.grid["B2"]),
                disk2.animate.move_to(self.grid["B4"]),
                disk2.animate.move_to(self.grid["F4"])
            ),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR),
            run_time=0.5
        )
        
        self.play(
            update_binary(3),
            Succession(
                disk1.animate.move_to(self.grid["B6"]),
                disk1.animate.move_to(self.grid["B4"]),
                disk1.animate.move_to(self.grid["E4"])
            ),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HIGHLIGHT_COLOR),
            run_time=0.5
        )
        
        self.play(
            update_binary(4),
            Succession(
                disk3.animate.move_to(self.grid["B2"]),
                disk3.animate.move_to(self.grid["B6"]),
                disk3.animate.move_to(self.grid["F6"])
            ),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HIGHLIGHT_COLOR),
            run_time=0.5
        )
        
        # Fast forward Step 5 (101): Disk 1 from Peg 2 to Peg 1
        self.play(
            update_binary(5),
            Succession(
                disk1.animate.move_to(self.grid["B4"]),
                disk1.animate.move_to(self.grid["B2"]),
                disk1.animate.move_to(self.grid["F2"])
            ),
            run_time=0.8
        )
        
        # Fast forward Step 6 (110): Disk 2 from Peg 2 to Peg 3
        self.play(
            update_binary(6),
            Succession(
                disk2.animate.move_to(self.grid["B4"]),
                disk2.animate.move_to(self.grid["B6"]),
                disk2.animate.move_to(self.grid["E6"])
            ),
            run_time=0.8
        )

        # Final Step 7: 111, Disk 1 from Peg 1 to Peg 3
        self.play(
            update_binary(7),
            Succession(
                disk1.animate.move_to(self.grid["B2"]),
                disk1.animate.move_to(self.grid["B6"]),
                disk1.animate.move_to(self.grid["D6"])
            ),
            run_time=1.5
        )
        self.wait(2)
