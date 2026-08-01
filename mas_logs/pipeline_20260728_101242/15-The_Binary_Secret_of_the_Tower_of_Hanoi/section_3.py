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
            "Map each disk to a specific binary bit.",
            "The smallest disk corresponds to the rightmost bit.",
            "Second disk maps to the second binary bit.",
            "The largest disk maps to the third bit.",
            "Move the disk when its bit changes value."
        ]
        self.setup_layout("The Mapping: Disks as Bits", lecture_lines)
        
        MAGENTA = "#FF00FF"
        DISK_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        PEG_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/peg.svg"
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(MAGENTA)
        
        # Disks and Labels using assets [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg]
        disk3 = SVGMobject(DISK_PATH).set_color(MAGENTA)
        disk2 = SVGMobject(DISK_PATH).set_color(MAGENTA)
        disk1 = SVGMobject(DISK_PATH).set_color(MAGENTA)
        
        # Scale to distinguish sizes (smallest to largest)
        disk3.scale(0.8) 
        disk2.scale(0.6)
        disk1.scale(0.4) 
        
        # Text labels for the disks
        d3_label = Text("Disk 3", font_size=20, color=MAGENTA)
        d2_label = Text("Disk 2", font_size=20, color=MAGENTA)
        d1_label = Text("Disk 1", font_size=20, color=MAGENTA)
        
        # Positioning according to Issue 48 (Labels in col 2, Disks in col 3)
        self.place_at_grid(d3_label, "A2")
        self.place_at_grid(disk3, "A3")
        self.place_at_grid(d2_label, "B2")
        self.place_at_grid(disk2, "B3")
        self.place_at_grid(d1_label, "C2")
        self.place_at_grid(disk1, "C3")
        
        # Bit power labels according to Issue 47 (Bit labels in col 4)
        bit3_label = MathTex("2^2", color=MAGENTA)
        bit2_label = MathTex("2^1", color=MAGENTA)
        bit1_label = MathTex("2^0", color=MAGENTA)
        
        self.place_at_grid(bit3_label, "A4")
        self.place_at_grid(bit2_label, "B4")
        self.place_at_grid(bit1_label, "C4")
        
        # Mapping arrows between disks and bits
        arrow3 = Arrow(disk3.get_right(), bit3_label.get_left(), color=MAGENTA, buff=0.1)
        arrow2 = Arrow(disk2.get_right(), bit2_label.get_left(), color=MAGENTA, buff=0.1)
        arrow1 = Arrow(disk1.get_right(), bit1_label.get_left(), color=MAGENTA, buff=0.1)
        
        self.play(
            FadeIn(disk1, disk2, disk3, d1_label, d2_label, d3_label),
            FadeIn(bit1_label, bit2_label, bit3_label),
            Create(arrow1), Create(arrow2), Create(arrow3)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(MAGENTA)
        
        # Highlight Disk 1 and Bit 2^0 (Rightmost)
        self.play(
            disk1.animate.set_color(MAGENTA), bit1_label.animate.set_color(MAGENTA), d1_label.animate.set_color(MAGENTA), arrow1.animate.set_color(MAGENTA),
            disk2.animate.set_color(WHITE), bit2_label.animate.set_color(WHITE), d2_label.animate.set_color(WHITE), arrow2.animate.set_color(WHITE),
            disk3.animate.set_color(WHITE), bit3_label.animate.set_color(WHITE), d3_label.animate.set_color(WHITE), arrow3.animate.set_color(WHITE),
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(MAGENTA)
        
        # Highlight Disk 2 and Bit 2^1
        self.play(
            disk1.animate.set_color(WHITE), bit1_label.animate.set_color(WHITE), d1_label.animate.set_color(WHITE), arrow1.animate.set_color(WHITE),
            disk2.animate.set_color(MAGENTA), bit2_label.animate.set_color(MAGENTA), d2_label.animate.set_color(MAGENTA), arrow2.animate.set_color(MAGENTA),
            disk3.animate.set_color(WHITE), bit3_label.animate.set_color(WHITE), d3_label.animate.set_color(WHITE), arrow3.animate.set_color(WHITE),
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(MAGENTA)
        
        # Highlight Disk 3 and Bit 2^2
        self.play(
            disk1.animate.set_color(WHITE), bit1_label.animate.set_color(WHITE), d1_label.animate.set_color(WHITE), arrow1.animate.set_color(WHITE),
            disk2.animate.set_color(WHITE), bit2_label.animate.set_color(WHITE), d2_label.animate.set_color(WHITE), arrow2.animate.set_color(WHITE),
            disk3.animate.set_color(MAGENTA), bit3_label.animate.set_color(MAGENTA), d3_label.animate.set_color(MAGENTA), arrow3.animate.set_color(MAGENTA),
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(MAGENTA)
        
        # Transition to showing the movement mechanism
        self.play(
            FadeOut(d1_label, d2_label, d3_label),
            FadeOut(arrow1, arrow2, arrow3),
            disk1.animate.set_color(WHITE),
            disk2.animate.set_color(WHITE),
            disk3.animate.set_color(WHITE),
            bit1_label.animate.set_color(WHITE),
            bit2_label.animate.set_color(WHITE),
            bit3_label.animate.set_color(WHITE),
        )
        
        # Setup Binary Counter digits
        bit3_val = Text("0", font_size=36).move_to(self.grid["D4"])
        bit2_val = Text("0", font_size=36).move_to(self.grid["D5"])
        bit1_val = Text("0", font_size=36).move_to(self.grid["D6"])
        
        counter_labels = VGroup(
            MathTex("2^2", font_size=24).next_to(bit3_val, UP),
            MathTex("2^1", font_size=24).next_to(bit2_val, UP),
            MathTex("2^0", font_size=24).next_to(bit1_val, UP)
        )

        # Setup Tower Pegs [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/peg.svg]
        pegA = SVGMobject(PEG_PATH).set_color(WHITE)
        pegB = SVGMobject(PEG_PATH).set_color(WHITE)
        pegC = SVGMobject(PEG_PATH).set_color(WHITE)
        peg_base = Line(self.grid["F1"], self.grid["F6"], color=WHITE)
        
        self.place_at_grid(pegA, "E2", scale_factor=0.6)
        self.place_at_grid(pegB, "E4", scale_factor=0.6)
        self.place_at_grid(pegC, "E6", scale_factor=0.6)

        # Display Counter and move Disks to Peg A
        self.play(
            Create(peg_base), FadeIn(pegA, pegB, pegC),
            FadeIn(bit1_val, bit2_val, bit3_val, counter_labels),
            FadeOut(bit1_label, bit2_label, bit3_label),
            disk3.animate.move_to(self.grid["E2"] + DOWN*0.2),
            disk2.animate.move_to(self.grid["E2"] + UP*0.1),
            disk1.animate.move_to(self.grid["E2"] + UP*0.4)
        )
        self.wait(1)
        
        # Flip the rightmost bit (0 -> 1)
        new_bit1_val = Text("1", font_size=36, color=MAGENTA).move_to(self.grid["D6"])
        
        self.play(
            Transform(bit1_val, new_bit1_val),
            disk1.animate.set_color(MAGENTA)
        )
        self.wait(0.5)
        
        # Move Disk 1 from Peg A (E2) to Peg C (E6)
        self.play(
            disk1.animate.move_to(self.grid["C2"]),
            run_time=0.4
        )
        self.play(
            disk1.animate.move_to(self.grid["C6"]),
            run_time=0.4
        )
        self.play(
            disk1.animate.move_to(self.grid["E6"] + DOWN*0.2),
            run_time=0.4
        )
        
        self.wait(2)
