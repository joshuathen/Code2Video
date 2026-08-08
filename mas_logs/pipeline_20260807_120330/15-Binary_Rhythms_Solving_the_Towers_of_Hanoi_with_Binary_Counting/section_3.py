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
        # Data from storyboard and outline
        title = "The Magic Mapping Rule"
        lecture_lines = [
            "Every binary change dictates which disk to move.",
            "Identify the rightmost bit that changed in the move.",
            "Bit position determines the disk number to move.",
            "Move 1 flips bit 1: move disk 1.",
            "Move 2 flips bit 2: move disk 2."
        ]
        self.setup_layout(title, lecture_lines)

        # Colors
        DISK_COLORS = {
            1: "#FF5733", # Orange-Red
            2: "#33FF57", # Green
            3: "#3357FF"  # Blue
        }
        HL_COLOR = "#FFFF00" # Yellow
        LINK_COLOR = "#00FFFF" # Cyan for glowing links

        # Assets
        disk_asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"

        # Helper to create binary bits
        def create_binary_group(bits_str):
            group = VGroup()
            for char in bits_str:
                group.add(Text(char, font="Monospace", font_size=42, color=WHITE))
            group.arrange(RIGHT, buff=0.6)
            return group

        # Helper to create disk icons with labels
        def create_disks():
            # Disk 1
            d1 = SVGMobject(disk_asset_path).set_color(DISK_COLORS[1]).scale(0.3)
            l1 = Text("Disk 1", font_size=16, color=DISK_COLORS[1]).next_to(d1, DOWN, buff=0.1)
            # Disk 2
            d2 = SVGMobject(disk_asset_path).set_color(DISK_COLORS[2]).scale(0.4)
            l2 = Text("Disk 2", font_size=16, color=DISK_COLORS[2]).next_to(d2, DOWN, buff=0.1)
            # Disk 3
            d3 = SVGMobject(disk_asset_path).set_color(DISK_COLORS[3]).scale(0.5)
            l3 = Text("Disk 3", font_size=16, color=DISK_COLORS[3]).next_to(d3, DOWN, buff=0.1)
            
            return VGroup(VGroup(d1, l1), VGroup(d2, l2), VGroup(d3, l3)).arrange(RIGHT, buff=0.4)

        # Initial Setup
        binary_counter = create_binary_group("000")
        self.place_in_area(binary_counter, "B2", "B5")
        
        # Resolve Issues #24 & #25: Place disks closer and aligned with binary bits
        disks = create_disks()
        self.place_in_area(disks, "D2", "D5", scale_factor=0.9)

        # Bit labels (Bit 3, Bit 2, Bit 1)
        bit_labels = VGroup(
            Text("Bit 3", font_size=14, color=GRAY).next_to(binary_counter[0], UP, buff=0.2),
            Text("Bit 2", font_size=14, color=GRAY).next_to(binary_counter[1], UP, buff=0.2),
            Text("Bit 1", font_size=14, color=GRAY).next_to(binary_counter[2], UP, buff=0.2)
        )

        # === Animation for Lecture Line 1 ===
        # "Every binary change dictates which disk to move."
        self.play(self.lecture[0].animate.set_color(HL_COLOR))
        self.play(
            FadeIn(binary_counter),
            FadeIn(disks),
            FadeIn(bit_labels)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Identify the rightmost bit that changed in the move."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HL_COLOR)
        )
        
        # Transition 000 -> 001 (Move 1)
        # 1st bit (index 2) changes
        move1_binary = create_binary_group("001")
        self.place_in_area(move1_binary, "B2", "B5")
        
        self.play(
            Transform(binary_counter[2], move1_binary[2]),
            binary_counter[2].animate.set_color(DISK_COLORS[1]),
            bit_labels[2].animate.set_color(DISK_COLORS[1])
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Bit position determines the disk number to move."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HL_COLOR)
        )
        
        # Create visual link between Bit 1 and Disk 1
        link1 = Line(binary_counter[2].get_bottom(), disks[0][0].get_top(), color=LINK_COLOR, stroke_width=4)
        link1_glow = link1.copy().set_stroke(width=10, opacity=0.3)
        
        self.play(
            Create(link1),
            FadeIn(link1_glow),
            disks[0].animate.scale(1.1)
        )
        self.play(Indicate(disks[0][0], color=DISK_COLORS[1]))
        self.wait(1)
        self.play(
            FadeOut(link1),
            FadeOut(link1_glow),
            disks[0].animate.scale(1/1.1),
            binary_counter[2].animate.set_color(WHITE),
            bit_labels[2].animate.set_color(GRAY)
        )

        # === Animation for Lecture Line 4 ===
        # "Move 1 flips bit 1: move disk 1."
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(HL_COLOR)
        )
        
        # Highlight Move 1 Logic
        self.play(
            binary_counter[2].animate.set_color(DISK_COLORS[1]),
            disks[0][0].animate.set_stroke(HL_COLOR, width=2)
        )
        self.wait(1)
        self.play(
            binary_counter[2].animate.set_color(WHITE),
            disks[0][0].animate.set_stroke(width=0)
        )

        # === Animation for Lecture Line 5 ===
        # "Move 2 flips bit 2: move disk 2."
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(HL_COLOR)
        )
        
        # Transition 001 -> 010 (Move 2)
        # Bit 2 flips 0->1, Bit 1 flips 1->0. Rightmost change (0 to 1) is Bit 2.
        move2_binary = create_binary_group("010")
        self.place_in_area(move2_binary, "B2", "B5")
        
        self.play(
            Transform(binary_counter[1], move2_binary[1]),
            Transform(binary_counter[2], move2_binary[2])
        )
        
        # Highlight Bit 2 and Link to Disk 2
        self.play(
            binary_counter[1].animate.set_color(DISK_COLORS[2]),
            bit_labels[1].animate.set_color(DISK_COLORS[2])
        )
        
        link2 = Line(binary_counter[1].get_bottom(), disks[1][0].get_top(), color=LINK_COLOR, stroke_width=4)
        link2_glow = link2.copy().set_stroke(width=10, opacity=0.3)
        
        self.play(
            Create(link2),
            FadeIn(link2_glow),
            disks[1].animate.scale(1.1)
        )
        self.play(Indicate(disks[1][0], color=DISK_COLORS[2]))
        
        self.wait(2)

        # Final Cleanup
        self.play(
            FadeOut(link2),
            FadeOut(link2_glow),
            disks[1].animate.scale(1/1.1),
            self.lecture[4].animate.set_color(WHITE)
        )
        self.wait(1)
