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
        # Initialize Scene with correct 5-line script
        lecture_lines = [
            'Each disk size corresponds to a specific binary bit.',
            'Bit one flips, so we move the smallest disk.',
            'Now bit two flips, signaling the second disk moves.',
            'Bit one flips again, returning focus to disk one.',
            'The rightmost flipping bit always identifies the moving disk.'
        ]
        self.setup_layout("The Perfect Mapping", lecture_lines)

        # Color definitions and Assets
        COLOR_D1 = "#00FFFF" # Cyan
        COLOR_D2 = "#FF00FF" # Magenta
        COLOR_D3 = "#FFFFFF" # White
        ASSET_DISK = "/mmfs1/data/home/jthen/Code2Video/assets/icon/disk.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))

        # Create Disk Representations using Icons
        d3 = SVGMobject(ASSET_DISK).set_color(COLOR_D3)
        d2 = SVGMobject(ASSET_DISK).set_color(COLOR_D3)
        d1 = SVGMobject(ASSET_DISK).set_color(COLOR_D3)

        # Centered layout: Col 3 (Disk 3), Col 4 (Disk 2), Col 5 (Disk 1)
        self.place_at_grid(d3, "C3", scale_factor=0.6)
        self.place_at_grid(d2, "C4", scale_factor=0.45)
        self.place_at_grid(d1, "C5", scale_factor=0.3)

        # Labels positioned above disks
        label3 = Text("Disk 3", font_size=18, color=WHITE)
        label2 = Text("Disk 2", font_size=18, color=WHITE)
        label1 = Text("Disk 1", font_size=18, color=WHITE)

        self.place_at_grid(label3, "B3")
        self.place_at_grid(label2, "B4")
        self.place_at_grid(label1, "B5")

        # Binary Bits (Initially 000) below disks
        bit3 = Text("0", font_size=40, color=WHITE)
        bit2 = Text("0", font_size=40, color=WHITE)
        bit1 = Text("0", font_size=40, color=WHITE)

        self.place_at_grid(bit3, "D3")
        self.place_at_grid(bit2, "D4")
        self.place_at_grid(bit1, "D5")

        self.play(
            FadeIn(d1), FadeIn(d2), FadeIn(d3),
            Write(label1), Write(label2), Write(label3)
        )
        self.play(Write(bit1), Write(bit2), Write(bit3))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_D1)
        )

        # Counter 000 -> 001: Bit 1 flips, Disk 1 glows Cyan
        new_bit1_v1 = Text("1", font_size=40, color=COLOR_D1).move_to(bit1)
        
        self.play(
            Transform(bit1, new_bit1_v1),
            Indicate(new_bit1_v1, color=COLOR_D1),
            Indicate(d1, color=COLOR_D1)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_D2)
        )

        # Counter 001 -> 010: Bit 2 flips, Disk 2 glows Magenta
        new_bit1_v2 = Text("0", font_size=40, color=WHITE).move_to(bit1)
        new_bit2_v1 = Text("1", font_size=40, color=COLOR_D2).move_to(bit2)

        self.play(
            Transform(bit1, new_bit1_v2),
            Transform(bit2, new_bit2_v1),
            Indicate(new_bit2_v1, color=COLOR_D2),
            Indicate(d2, color=COLOR_D2)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(COLOR_D1)
        )

        # Counter 010 -> 011: Bit 1 flips again, Disk 1 glows Cyan
        new_bit1_v3 = Text("1", font_size=40, color=COLOR_D1).move_to(bit1)

        self.play(
            Transform(bit1, new_bit1_v3),
            Indicate(new_bit1_v3, color=COLOR_D1),
            Indicate(d1, color=COLOR_D1)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(YELLOW)
        )

        # Display final rule summary
        summary_text = Text("Rightmost bit flip = Moving Disk", color=WHITE, font_size=24)
        self.place_in_area(summary_text, 'E2', 'F5')
        self.play(Write(summary_text))
        self.wait(2)
