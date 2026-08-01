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
        # Data
        title = "The Directional Logic"
        lines = [
            "Imagine the pegs arranged in a clockwise circle.",
            "Disk one always follows this strict clockwise path.",
            "Other disks simply move to the only legal spot."
        ]
        
        self.setup_layout(title, lines)

        # Colors
        COLOR_1 = "#FFFF00" # Yellow
        COLOR_2 = "#00FFFF" # Cyan
        COLOR_3 = "#00FF00" # Green

        # Asset paths
        PEG_ASSET = "/mmfs1/data/home/jthen/Code2Video/assets/icon/peg.svg"
        DISK_ASSET = "/mmfs1/data/home/jthen/Code2Video/assets/icon/disk.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(COLOR_1))
        
        # Pegs (using assets)
        peg_a = SVGMobject(PEG_ASSET).set_color(GREY_B)
        peg_b = SVGMobject(PEG_ASSET).set_color(GREY_B)
        peg_c = SVGMobject(PEG_ASSET).set_color(GREY_B)
        
        self.place_at_grid(peg_a, "B4", scale_factor=0.6)
        self.place_at_grid(peg_b, "E2", scale_factor=0.6)
        self.place_at_grid(peg_c, "E6", scale_factor=0.6)
        
        label_a = Text("Peg A", font_size=18)
        label_b = Text("Peg B", font_size=18)
        label_c = Text("Peg C", font_size=18)
        
        self.place_at_grid(label_a, "A4", scale_factor=1.0)
        self.place_at_grid(label_b, "F2", scale_factor=1.0)
        self.place_at_grid(label_c, "F6", scale_factor=1.0)

        # Arrows (connecting the grid centers in a circle)
        arrow_ac = CurvedArrow(self.grid["B5"], self.grid["D6"], angle=-TAU/6, color=WHITE)
        arrow_cb = CurvedArrow(self.grid["E5"], self.grid["E3"], angle=-TAU/6, color=WHITE)
        arrow_ba = CurvedArrow(self.grid["D2"], self.grid["C3"], angle=-TAU/6, color=WHITE)

        self.play(
            FadeIn(peg_a), FadeIn(peg_b), FadeIn(peg_c),
            Write(label_a), Write(label_b), Write(label_c)
        )
        self.play(Create(arrow_ac), Create(arrow_cb), Create(arrow_ba))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(COLOR_2))
        
        # Disk 1 (using asset)
        disk1 = SVGMobject(DISK_ASSET).set_color(COLOR_1)
        self.place_at_grid(disk1, "B4", scale_factor=0.4)
        
        self.play(FadeIn(disk1))
        # A -> C
        self.play(disk1.animate.move_to(self.grid["E6"]))
        self.wait(0.5)
        # C -> B
        self.play(disk1.animate.move_to(self.grid["E2"]))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(COLOR_3))
        
        # Show Disk 2 at A
        disk2 = SVGMobject(DISK_ASSET).set_color(COLOR_2)
        self.place_at_grid(disk2, "B4", scale_factor=0.6)
        self.play(FadeIn(disk2))
        
        # Highlight legal move for Disk 2: B has Disk 1, so move to C.
        flash_rect = Square(color=COLOR_2, stroke_width=2).scale(0.4)
        self.place_at_grid(flash_rect, "E6")
        
        self.play(Indicate(peg_c, color=COLOR_2), Flash(self.grid["E6"], color=COLOR_2))
        # Move Disk 2 to C
        self.play(disk2.animate.move_to(self.grid["E6"]))
        
        # Rhythmic cycle: now Disk 1 moves to next spot (B -> A)
        self.play(disk1.animate.move_to(self.grid["B4"]))
        
        self.wait(2)
