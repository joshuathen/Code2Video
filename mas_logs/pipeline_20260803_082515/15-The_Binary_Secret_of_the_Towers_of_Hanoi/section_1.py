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
        # Assets
        PEG_SVG = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/peg.svg"
        DISK_SVG = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/disk.svg"

        self.setup_layout("The Legend and the Rules", [
            "Welcome to the legendary Towers of Hanoi puzzle.",
            "We have three pegs and stacked disks.",
            "Move one disk; never place larger on smaller."
        ])

        # Define colors
        COLOR_GOLD = "#FFD700"
        COLOR_RED = "#FF0000"
        COLOR_GREEN = "#00FF00"
        COLOR_BLUE = "#0000FF"
        COLOR_WHITE = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Show three pegs (labeled A, B, C) and 3 disks (Red, Green, Blue) on Peg A.
        self.lecture[0].set_color(YELLOW)
        
        # Use SVGMobjects for pegs
        peg_a = SVGMobject(PEG_SVG).set_color(COLOR_WHITE)
        peg_b = SVGMobject(PEG_SVG).set_color(COLOR_WHITE)
        peg_c = SVGMobject(PEG_SVG).set_color(COLOR_WHITE)
        
        # Issue 28: Positioning pegs at C2, C4, C6
        self.place_at_grid(peg_a, "C2", scale_factor=1.0)
        self.place_at_grid(peg_b, "C4", scale_factor=1.0)
        self.place_at_grid(peg_c, "C6", scale_factor=1.0)
        
        # Labels at D2, D4, D6 with scale_factor=1.2
        label_a = Text("A", font_size=20, color=COLOR_WHITE)
        label_b = Text("B", font_size=20, color=COLOR_WHITE)
        label_c = Text("C", font_size=20, color=COLOR_WHITE)
        
        self.place_at_grid(label_a, "D2", scale_factor=1.2)
        self.place_at_grid(label_b, "D4", scale_factor=1.2)
        self.place_at_grid(label_c, "D6", scale_factor=1.2)
        
        # Disks as SVGMobjects
        # We'll adjust their widths to differentiate sizes
        disk_blue = SVGMobject(DISK_SVG).set_color(COLOR_BLUE).set_stroke(COLOR_BLUE)
        disk_green = SVGMobject(DISK_SVG).set_color(COLOR_GREEN).set_stroke(COLOR_GREEN)
        disk_red = SVGMobject(DISK_SVG).set_color(COLOR_RED).set_stroke(COLOR_RED)
        
        # Set individual scales for sizes
        disk_blue.scale(0.8) # Largest
        disk_green.scale(0.6) # Medium
        disk_red.scale(0.4) # Smallest
        
        # Stacking on Peg A (C2)
        # Assuming peg height and disk height are consistent. 
        # C2 center is (1.5, 0.2). 
        base_y = self.grid["C2"][1] - 0.8
        disk_blue.move_to([self.grid["C2"][0], base_y, 0])
        disk_green.move_to([self.grid["C2"][0], base_y + 0.35, 0])
        disk_red.move_to([self.grid["C2"][0], base_y + 0.7, 0])
        
        pegs = VGroup(peg_a, peg_b, peg_c)
        labels = VGroup(label_a, label_b, label_c)
        disks = VGroup(disk_blue, disk_green, disk_red)
        
        self.play(FadeIn(pegs), FadeIn(labels), FadeIn(disks))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight the three pegs with #FFD700 (Gold) glow.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        glows = VGroup(*[
            peg.copy().set_stroke(COLOR_GOLD, 8).set_fill(opacity=0)
            for peg in pegs
        ])
        
        self.play(Create(glows))
        self.play(FadeOut(glows))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Move smallest disk (Red) A -> B. Then Green A -> B (fail).
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # 1. Red disk moves A -> B
        target_b_pos = [self.grid["C4"][0], base_y, 0] # Bottom of Peg B
        self.play(disk_red.animate.move_to(target_b_pos))
        self.wait(0.5)
        
        # 2. Green disk moves A -> B (attempt)
        illegal_pos = [self.grid["C4"][0], base_y + 0.35, 0] # Above Red on B
        self.play(disk_green.animate.move_to(illegal_pos))
        
        # 3. Show red 'X'
        cross = VGroup(
            Line(UP+LEFT, DOWN+RIGHT, color=COLOR_RED, stroke_width=8),
            Line(UP+RIGHT, DOWN+LEFT, color=COLOR_RED, stroke_width=8)
        ).scale(0.3).move_to(illegal_pos + UP * 0.5)
        
        self.play(Create(cross))
        self.play(Flash(cross, color=COLOR_RED))
        self.wait(1)
        
        # 4. Return disks to A for goal setup
        self.play(
            FadeOut(cross), 
            disk_green.animate.move_to([self.grid["C2"][0], base_y + 0.35, 0]),
            disk_red.animate.move_to([self.grid["C2"][0], base_y + 0.7, 0])
        )
        self.wait(0.5)
        
        # 5. Goal: Move all to C
        # Final positions on C
        final_blue = [self.grid["C6"][0], base_y, 0]
        final_green = [self.grid["C6"][0], base_y + 0.35, 0]
        final_red = [self.grid["C6"][0], base_y + 0.7, 0]
        
        self.play(
            disk_blue.animate.move_to(final_blue),
            disk_green.animate.move_to(final_green),
            disk_red.animate.move_to(final_red),
            run_time=2
        )
        
        # Flash entire completed stack on Peg C with #00FF00 (Green)
        self.play(
            disks.animate.set_color(COLOR_GREEN).set_stroke(COLOR_GREEN),
            Flash(self.grid["C6"], color=COLOR_GREEN, flash_radius=1.5)
        )
        self.wait(2)
        
        self.lecture[2].set_color(WHITE)
