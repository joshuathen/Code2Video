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
        # Data from storyboard
        title = "The Puzzle Setup: Rules of the Game"
        lines = [
            "The Tower of Hanoi has three pegs and disks.",
            "Rule one: move only one disk at a time.",
            "Rule two: no larger disk atop a smaller one.",
            "Goal: move the stack to a different peg.",
            "Binary counting reveals the most efficient solution."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_DISK1 = BLUE_B
        COLOR_DISK2 = GREEN_B
        COLOR_DISK3 = ORANGE
        COLOR_PEG = GRAY_B
        COLOR_HIGHLIGHT = YELLOW
        COLOR_INVALID = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # "The Tower of Hanoi has three pegs and disks."
        self.lecture[0].set_color(YELLOW)
        
        # Pegs - using grid points for alignment (Issue 33 fix)
        peg_h = 3.2
        peg_a = Rectangle(width=0.1, height=peg_h, fill_opacity=1, color=COLOR_PEG, stroke_width=0)
        peg_b = Rectangle(width=0.1, height=peg_h, fill_opacity=1, color=COLOR_PEG, stroke_width=0)
        peg_c = Rectangle(width=0.1, height=peg_h, fill_opacity=1, color=COLOR_PEG, stroke_width=0)
        
        self.place_at_grid(peg_a, "D2")
        self.place_at_grid(peg_b, "D4")
        self.place_at_grid(peg_c, "D6")
        
        labels = VGroup(
            Text("A", font_size=24),
            Text("B", font_size=24),
            Text("C", font_size=24)
        )
        labels[0].next_to(self.grid["F2"], DOWN, buff=0.1)
        labels[1].next_to(self.grid["F4"], DOWN, buff=0.1)
        labels[2].next_to(self.grid["F6"], DOWN, buff=0.1)

        # Disks
        disk_h = 0.5
        disk3 = RoundedRectangle(corner_radius=0.1, width=1.8, height=disk_h, fill_opacity=1, color=COLOR_DISK3)
        disk2 = RoundedRectangle(corner_radius=0.1, width=1.3, height=disk_h, fill_opacity=1, color=COLOR_DISK2)
        disk1 = RoundedRectangle(corner_radius=0.1, width=0.8, height=disk_h, fill_opacity=1, color=COLOR_DISK1)
        
        # Stack disks on Peg A using grid slots (Issue 34 fix)
        self.place_at_grid(disk3, "E2")
        self.place_at_grid(disk2, "D2")
        self.place_at_grid(disk1, "C2")

        self.play(FadeIn(VGroup(peg_a, peg_b, peg_c)), Write(labels), FadeIn(VGroup(disk1, disk2, disk3)))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Rule one: move only one disk at a time."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Animate top disk (Disk 1) moving from Peg A (C2) to Peg B (E4)
        self.play(disk1.animate.move_to(self.grid["B2"]), run_time=0.4)
        self.play(disk1.animate.move_to(self.grid["B4"]), run_time=0.4)
        self.play(disk1.animate.move_to(self.grid["E4"]), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Rule two: no larger disk atop a smaller one."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Try to move Disk 2 on top of Disk 1 at Peg B (E4 -> D4)
        self.play(disk2.animate.move_to(self.grid["B2"]), run_time=0.4)
        self.play(disk2.animate.move_to(self.grid["B4"]), run_time=0.4)
        self.play(disk2.animate.move_to(self.grid["D4"]), run_time=0.4)
        
        # Flash Red indicating invalid move
        flash_rect = SurroundingRectangle(disk2, color=COLOR_INVALID, buff=0.1)
        self.play(Create(flash_rect))
        self.play(disk2.animate.set_fill(COLOR_INVALID), run_time=0.2)
        self.play(disk2.animate.set_fill(COLOR_DISK2), run_time=0.2)
        self.play(FadeOut(flash_rect))
        
        # Return Disk 2 to Peg A (D2)
        self.play(disk2.animate.move_to(self.grid["B4"]), run_time=0.3)
        self.play(disk2.animate.move_to(self.grid["B2"]), run_time=0.3)
        self.play(disk2.animate.move_to(self.grid["D2"]), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Goal: move the stack to a different peg."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Move Disk 2 to Peg C (E6)
        self.play(disk2.animate.move_to(self.grid["B2"]), run_time=0.4)
        self.play(disk2.animate.move_to(self.grid["B6"]), run_time=0.4)
        self.play(disk2.animate.move_to(self.grid["E6"]), run_time=0.4)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Binary counting reveals the most efficient solution."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Highlight entire stack on A (Disk 3) and target Peg C
        highlight_a = SurroundingRectangle(VGroup(peg_a, disk3), color=COLOR_HIGHLIGHT, buff=0.2)
        highlight_c = SurroundingRectangle(VGroup(peg_c, disk2), color=COLOR_HIGHLIGHT, buff=0.2)
        
        self.play(Create(highlight_a), Create(highlight_c))
        self.wait(2)
        self.play(FadeOut(highlight_a), FadeOut(highlight_c))
        self.wait(1)
