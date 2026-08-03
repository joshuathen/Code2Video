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
        # Setup section titles and lecture lines
        title_text = "The Hidden Connection: Move = Bit Position"
        lecture_lines = [
            "Position one is disk one; position two is disk two.",
            "Move four uses bit three for the largest disk.",
            "Binary digits guide every move without complex recursion."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Constants
        HIGHLIGHT_COLOR = "#FF8C00" # Orange specified in storyboard
        DISK_COLORS = ["#88C0D0", "#A3BE8C", "#EBCB8B"] # Small, Med, Large
        DISK_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"
        PEG_PATH = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/peg.svg"

        # === Animation for Lecture Line 1 ===
        # Split screen: Left side shows text 'Move 2: 010'. Right side shows 3 disks on pegs.
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        binary_label = Text("Move 2:", font_size=32)
        binary_val = Text("010", font_size=32)
        binary_group = VGroup(binary_label, binary_val).arrange(RIGHT, buff=0.2)
        # Fix for issue 30: Adjust position and scale
        self.place_in_area(binary_group, "B2", "D3", scale_factor=0.8)
        
        # Create Pegs using SVGMobject
        pegs = VGroup(*[SVGMobject(PEG_PATH).set_color(GRAY) for _ in range(3)])
        pegs.arrange(RIGHT, buff=1.2)
        # Fix for issue 31: Adjust scale
        self.place_in_area(pegs, "B4", "E6", scale_factor=0.9)
        
        # Create Disks using SVGMobject
        # We vary the width by scaling the SVG
        disk_scales = [0.4, 0.6, 0.8]
        disks = VGroup()
        for i in range(3):
            d = SVGMobject(DISK_PATH).set_color(DISK_COLORS[i])
            d.scale(disk_scales[i])
            disks.add(d)
        
        # Position disks on the first peg (leftmost)
        # Smallest is index 0, largest is index 2.
        peg_base = pegs[0].get_bottom() + UP * 0.2
        disks[2].move_to(peg_base)
        disks[1].move_to(peg_base + UP * 0.35)
        disks[0].move_to(peg_base + UP * 0.7)
        
        self.play(
            FadeIn(binary_group),
            FadeIn(pegs),
            FadeIn(disks)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # In the binary string '010', the second bit (position 2) glows #FF8C00 (Orange).
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        # '010' has '1' at index 1.
        bit_to_glow = binary_val[1]
        glow_rect = SurroundingRectangle(bit_to_glow, color=HIGHLIGHT_COLOR, buff=0.1)
        
        self.play(
            Create(glow_rect),
            bit_to_glow.animate.set_color(HIGHLIGHT_COLOR)
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # On the right, Disk 2 (the medium one) glows #FF8C00 (Orange) and lifts slightly.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR)
        )
        
        disk2 = disks[1] # Medium disk is index 1
        
        self.play(
            disk2.animate.set_color(HIGHLIGHT_COLOR).shift(UP * 0.6),
            glow_rect.animate.scale(1.2).set_stroke(width=4)
        )
        self.wait(2)
        
        # Final cleanup/reset for smooth transition
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            disk2.animate.set_color(DISK_COLORS[1]).shift(DOWN * 0.6),
            FadeOut(glow_rect),
            bit_to_glow.animate.set_color(WHITE)
        )
        self.wait(1)
