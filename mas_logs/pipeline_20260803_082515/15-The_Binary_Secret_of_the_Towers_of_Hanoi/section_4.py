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
        # Lecture lines from storyboard
        lecture_lines = [
            "Smallest disk always follows a clockwise circular path.",
            "For larger disks, only one legal move is possible.",
            "Binary tells you when, rules tell you where."
        ]
        
        self.setup_layout("The Directional Logic: Odd vs. Even", lecture_lines)
        
        # Define Colors
        DISK1_COLOR = RED
        DISK2_COLOR = GREEN
        TRAIL_COLOR = "#ADD8E6"
        ARROW_COLOR = "#00FF00"

        # Pegs (Base for visual context)
        peg_a_base = Line(self.grid["D2"] + LEFT*0.4, self.grid["D2"] + RIGHT*0.4, color=GRAY)
        peg_b_base = Line(self.grid["D4"] + LEFT*0.4, self.grid["D4"] + RIGHT*0.4, color=GRAY)
        peg_c_base = Line(self.grid["D6"] + LEFT*0.4, self.grid["D6"] + RIGHT*0.4, color=GRAY)
        peg_a_stem = Line(self.grid["D2"], self.grid["B2"], color=GRAY)
        peg_b_stem = Line(self.grid["D4"], self.grid["B4"], color=GRAY)
        peg_c_stem = Line(self.grid["D6"], self.grid["B6"], color=GRAY)
        pegs = VGroup(peg_a_base, peg_b_base, peg_c_base, peg_a_stem, peg_b_stem, peg_c_stem)
        self.add(pegs)

        # Labels for Pegs
        label_a = Text("A", font_size=20).next_to(peg_a_base, DOWN, buff=0.1)
        label_b = Text("B", font_size=20).next_to(peg_b_base, DOWN, buff=0.1)
        label_c = Text("C", font_size=20).next_to(peg_c_base, DOWN, buff=0.1)
        self.add(label_a, label_b, label_c)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(DISK1_COLOR))

        # Disk 1 (Red)
        disk1 = RoundedRectangle(corner_radius=0.1, height=0.2, width=0.6, fill_opacity=1, color=DISK1_COLOR)
        # Fix for Issue 32: Starting disk1 at B4 (Peg B) to avoid initial overlap with disk2
        self.place_at_grid(disk1, "B4", scale_factor=0.8)
        
        # Trail
        trail = TracedPath(disk1.get_center, stroke_color=TRAIL_COLOR, stroke_width=4, dissipating_time=0.5)
        self.add(trail)
        self.add(disk1)

        # Cyclic path: B -> C -> A -> B
        path_points = ["B6", "B2", "B4"]
        for pos in path_points:
            self.play(disk1.animate.move_to(self.grid[pos]), run_time=0.8, rate_func=linear)
        
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(DISK2_COLOR))

        # Ensure Disk 1 is at Peg B (B4) to leave Peg A clear for Disk 2
        # (It already returned to B4 in the cycle above)

        # Disk 2 (Green)
        disk2 = RoundedRectangle(corner_radius=0.1, height=0.2, width=1.0, fill_opacity=1, color=DISK2_COLOR)
        # Initialize at C2 (Peg A)
        self.place_at_grid(disk2, "C2", scale_factor=0.8)
        self.add(disk2)
        
        # Highlight legal move for Disk 2: A to C (since B is occupied by Disk 1)
        arrow = Arrow(start=self.grid["C2"], end=self.grid["C6"], color=ARROW_COLOR, buff=0.3)
        self.play(GrowArrow(arrow))
        
        # Move Disk 2 to C6 (Peg C) - Addresses intended destination in Issue 33
        self.play(disk2.animate.move_to(self.grid["C6"]), run_time=1.5)
        self.play(FadeOut(arrow))

        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))

        # 'Fast' label for Disk 1
        fast_label = Text("Fast", font_size=18, color=YELLOW)
        fast_label.next_to(disk1, UP, buff=0.1)
        
        # 'Slow' label for Disk 2
        slow_label = Text("Slow", font_size=18, color=YELLOW)
        slow_label.next_to(disk2, UP, buff=0.1)

        self.play(Write(fast_label), Write(slow_label))
        
        # Small visual emphasis for different speeds
        self.play(
            disk1.animate.shift(UP * 0.2).set_rate_func(rate_functions.there_and_back),
            run_time=0.4
        )
        self.play(
            disk1.animate.shift(UP * 0.2).set_rate_func(rate_functions.there_and_back),
            run_time=0.4
        )

        self.wait(2)
