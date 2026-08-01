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
        self.setup_layout(
            "The Legend and the Rules", 
            [
                "Meet the legendary puzzle: three pegs and graduated disks.",
                "Move the smallest disk from the left to right.",
                "Then, move the medium disk to the middle peg.",
                "Important: never place a larger disk onto smaller ones.",
                "The goal: rebuild the stack on a different peg."
            ]
        )
        
        # Define colors
        COLOR_PEG = "#808080"
        COLOR_DISK1 = "#FF0000" # Red (Smallest)
        COLOR_DISK2 = "#00FF00" # Green (Medium)
        COLOR_DISK3 = "#0000FF" # Blue (Largest)
        COLOR_X = "#FF0000"
        COLOR_GOLD = "#FFD700"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Create Pegs (Labels in Row A, Pegs B-F)
        peg_a = Line(self.grid["B2"], self.grid["F2"], color=COLOR_PEG, stroke_width=8)
        peg_b = Line(self.grid["B4"], self.grid["F4"], color=COLOR_PEG, stroke_width=8)
        peg_c = Line(self.grid["B6"], self.grid["F6"], color=COLOR_PEG, stroke_width=8)
        
        label_a = Text("A", font_size=24)
        label_b = Text("B", font_size=24)
        label_c = Text("C", font_size=24)
        
        self.place_at_grid(label_a, "A2")
        self.place_at_grid(label_b, "A4")
        self.place_at_grid(label_c, "A6")
        
        # Create Disks
        disk_blue = RoundedRectangle(width=1.6, height=0.4, corner_radius=0.1, fill_color=COLOR_DISK3, fill_opacity=1, stroke_width=2)
        disk_green = RoundedRectangle(width=1.2, height=0.4, corner_radius=0.1, fill_color=COLOR_DISK2, fill_opacity=1, stroke_width=2)
        disk_red = RoundedRectangle(width=0.8, height=0.4, corner_radius=0.1, fill_color=COLOR_DISK1, fill_opacity=1, stroke_width=2)
        
        # Stack disks on Peg A using updated grid positions (F2 bottom, then E2, then D2)
        self.place_at_grid(disk_blue, "F2", scale_factor=1.3)
        self.place_at_grid(disk_green, "E2", scale_factor=1.3)
        self.place_at_grid(disk_red, "D2", scale_factor=1.3)
        
        self.play(Create(peg_a), Create(peg_b), Create(peg_c), Write(label_a), Write(label_b), Write(label_c))
        self.play(FadeIn(disk_blue), FadeIn(disk_green), FadeIn(disk_red))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))
        
        # Parabolic move Red: Peg A (D2) to Peg C (F6)
        path_red = ArcBetweenPoints(start=self.grid["D2"], end=self.grid["F6"], angle=-TAU/4)
        self.play(MoveAlongPath(disk_red, path_red), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))
        
        # Move Green: Peg A (E2) to Peg B (F4)
        path_green = ArcBetweenPoints(start=self.grid["E2"], end=self.grid["F4"], angle=-TAU/4)
        self.play(MoveAlongPath(disk_green, path_green), run_time=1.2)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))
        
        # Attempt to move Blue: Peg A (F2) halfway to Peg B (above Green at E4)
        target_illegal = self.grid["E4"]
        midpoint = (self.grid["F2"] + target_illegal) / 2 + UP * 0.8
        
        self.play(disk_blue.animate.move_to(midpoint), run_time=0.8)
        
        # Show flashing Red 'X'
        cross = VGroup(
            Line(midpoint + LEFT*0.4 + UP*0.4, midpoint + RIGHT*0.4 + DOWN*0.4, color=COLOR_X, stroke_width=12),
            Line(midpoint + LEFT*0.4 + DOWN*0.4, midpoint + RIGHT*0.4 + UP*0.4, color=COLOR_X, stroke_width=12)
        )
        self.play(FadeIn(cross))
        self.play(Flash(cross, color=COLOR_X, flash_radius=0.6))
        self.play(FadeOut(cross))
        
        # Return Blue to Peg A (F2)
        self.play(disk_blue.animate.move_to(self.grid["F2"]), run_time=0.8)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))
        
        # Instant transition to target stack on Peg C
        target_blue = disk_blue.copy().move_to(self.grid["F6"])
        target_green = disk_green.copy().move_to(self.grid["E6"])
        target_red = disk_red.copy().move_to(self.grid["D6"])
        
        target_stack = VGroup(target_blue, target_green, target_red)
        
        self.play(
            FadeOut(disk_red),
            FadeOut(disk_green),
            FadeOut(disk_blue),
            FadeIn(target_stack)
        )
        
        # Gold Glow effect
        glow = target_stack.copy().set_color(COLOR_GOLD).set_opacity(0.3).scale(1.05)
        self.play(
            target_stack.animate.set_stroke(COLOR_GOLD, 5),
            FadeIn(glow),
            rate_func=there_and_back,
            run_time=2
        )
        self.play(FadeOut(glow))
        
        self.wait(2)
