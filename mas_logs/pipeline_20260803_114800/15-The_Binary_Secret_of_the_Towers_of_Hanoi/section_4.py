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
        lecture_lines = [
            "To find the disk, write the move number in binary.",
            "Find the position of the rightmost digit one.",
            "Position one means move the smallest disk.",
            "Position two means move the second smallest disk.",
            "This pattern holds for every move in the puzzle."
        ]
        self.setup_layout("The Rule of the Rightmost Set Bit", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show the decimal number '4' and its binary equivalent '100'
        # Issue 26: Shift binary conversion one column to the right (B4, B5, B6)
        decimal_4 = Text("4", font_size=48, color=WHITE)
        arrow_to_bin = MathTex("\\rightarrow", color=WHITE)
        binary_4 = MathTex("1", "0", "0", font_size=48, color=WHITE)
        
        self.place_at_grid(decimal_4, "B4", scale_factor=1.0)
        self.place_at_grid(arrow_to_bin, "B5", scale_factor=1.0)
        self.place_at_grid(binary_4, "B6", scale_factor=1.0)
        
        self.lecture[0].set_color(YELLOW)
        self.play(Write(decimal_4), Write(arrow_to_bin), Write(binary_4))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Underline the '1' in '100' with a gold color (#FFD700)
        # Issue 27: Realign and scale the label to stay directly under the binary number at column 6.
        underline = Underline(binary_4[0], color="#FFD700", buff=0.1)
        pos_label = Text("Rightmost 1\nat position 3", font_size=20, color="#FFD700")
        self.place_at_grid(pos_label, "C6", scale_factor=0.8)

        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(Create(underline))
        self.play(FadeIn(pos_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Position one means move the smallest disk.
        # Issue 25: Move disks up to Row D.
        # Issue 20: Use [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg]
        disk1_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg").set_color("#1E90FF")
        disk1_label = Text("Disk 1", font_size=16, color=WHITE)
        disk1_group = VGroup(disk1_svg, disk1_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(disk1_group, "D4", scale_factor=0.4)
        
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        self.play(FadeIn(disk1_group))
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        # Position two means move the second smallest disk.
        # Issue 25: Move disks up to Row D.
        disk2_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg").set_color("#1E90FF")
        disk2_label = Text("Disk 2", font_size=16, color=WHITE)
        disk2_group = VGroup(disk2_svg, disk2_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(disk2_group, "D5", scale_factor=0.6)

        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(FadeIn(disk2_group))
        self.wait(0.5)

        # === Animation for Lecture Line 5 ===
        # This pattern holds for every move in the puzzle.
        # Issue 20: Use [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg] for Disk 3.
        # Issue 25: Move disks up to Row D.
        disk3_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg").set_color("#1E90FF")
        disk3_label = Text("Disk 3", font_size=16, color=WHITE)
        disk3_group = VGroup(disk3_svg, disk3_label).arrange(DOWN, buff=0.1)
        self.place_at_grid(disk3_group, "D6", scale_factor=0.8)

        # Arrow from the underlined '1' to Disk 3
        rule_arrow = Arrow(
            start=underline.get_bottom(),
            end=disk3_group.get_top(),
            color="#FFD700",
            stroke_width=3,
            buff=0.1
        )

        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        self.play(FadeIn(disk3_group))
        self.play(Create(rule_arrow))
        self.wait(2)
