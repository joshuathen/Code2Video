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
        self.setup_layout("Towers of Hanoi and Binary Logic", [
            "Moving N disks requires two to the N minus one moves.",
            "Seven moves are needed for three disks.",
            "Sequence counts binary like a rhythmic clock."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Draw three vertical stacks of varying sizes (#FF0000)
        disk1 = Rectangle(width=1.0, height=0.2, color=RED, fill_opacity=1)
        disk2 = Rectangle(width=1.5, height=0.2, color=RED, fill_opacity=1)
        disk3 = Rectangle(width=2.0, height=0.2, color=RED, fill_opacity=1)
        stack = VGroup(disk3, disk2, disk1).arrange(UP, buff=0.05)
        # Fix: Line 60 constraint
        self.place_at_grid(stack, "E5", scale_factor=0.6)
        self.play(FadeIn(stack))
        self.lecture[0].set_color("#FF0000")

        # === Animation for Lecture Line 2 ===
        # Place a numeric binary label (#00FFFF) beside each stack
        # Asset import: /scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg
        clock_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/clock.svg")
        label = Text("2^3 - 1 = 7", font_size=32, color="#00FFFF")
        # Fix: Line 67 constraint
        self.place_at_grid(label, "A3", scale_factor=0.9)
        self.place_at_grid(clock_icon, "A5", scale_factor=0.7)
        self.play(Write(label), FadeIn(clock_icon))
        self.lecture[1].set_color("#00FFFF")

        # === Animation for Lecture Line 3 ===
        # Animate the stacks transforming into a binary sequence (#FFFFFF)
        binary_seq = Text("1 - 2 - 1 - 3 - 1 - 2 - 1", font_size=24, color=WHITE)
        # Fix: Line 74 constraint
        self.place_in_area(binary_seq, "C3", "C6", scale_factor=0.7)
        self.play(ReplacementTransform(stack, binary_seq))
        self.lecture[2].set_color("#FFFFFF")
        self.wait(2)
