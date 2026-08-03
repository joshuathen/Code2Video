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

class Section6Scene(TeachingScene):
    def construct(self):
        self.setup_layout(
            "Conclusion: The Final Tally", 
            [
                "Larger mass ratios create smaller, more precise angles.",
                "The collision count perfectly matches the digits of Pi.",
                "A surprising link between simple physics and mathematical constants."
            ]
        )
        
        # Colors
        yellow_color = "#FFFF00"
        white_color = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(yellow_color)
        
        # Block simulation visuals
        wall = Line(UP, DOWN, color=white_color).scale(2)
        self.place_at_grid(wall, "C1")
        
        # Asset integration: Using the block SVG [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg]
        block_m = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        block_m.set_color(BLUE)
        block_M = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/block.svg")
        block_M.set_color(RED)
        
        self.place_at_grid(block_m, "C2", scale_factor=0.4)
        self.place_at_grid(block_M, "C3", scale_factor=0.8)
        
        counter_label = Text("Collisions:", font_size=24, color=white_color)
        counter_value = Text("3141", font_size=32, color=yellow_color)
        counter_group = VGroup(counter_label, counter_value).arrange(RIGHT, buff=0.3)
        
        # Fix for Issue 44: Positioning counter_group at B4 to reduce cramping
        self.place_at_grid(counter_group, "B4", scale_factor=1.1)
        
        self.play(
            FadeIn(wall),
            FadeIn(block_m),
            FadeIn(block_M),
            FadeIn(counter_group)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(white_color)
        self.lecture[1].set_color(yellow_color)
        
        # Table of mass ratios and counts
        table_header = Text("Mass Ratio | Bounces", font_size=20, color=white_color)
        table_row1 = Text("1 : 1          | 3", font_size=20, color=white_color)
        table_row2 = Text("1 : 100        | 31", font_size=20, color=white_color)
        table_row3 = Text("1 : 10,000     | 314", font_size=20, color=white_color)
        table_row4 = Text("1 : 1,000,000  | 3141", font_size=20, color=white_color)
        
        table = VGroup(table_header, table_row1, table_row2, table_row3, table_row4).arrange(DOWN, aligned_edge=LEFT, buff=0.2)
        
        # Fix for Issue 45: Positioning table in area E2-F6 for more breathing room
        self.place_in_area(table, "E2", "F6", scale_factor=0.85)
        
        self.play(Write(table))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(white_color)
        self.lecture[2].set_color(yellow_color)
        
        # Final transformation: fade out blocks/table and show Pi
        pi_text = Text("π ≈ 3.141...", font_size=48, color=yellow_color)
        
        # Fix for Issue 43: Positioning pi_text in area B2-D6 to avoid overlap with previous elements
        self.place_in_area(pi_text, "B2", "D6", scale_factor=1.4)
        
        self.play(
            FadeOut(wall),
            FadeOut(block_m),
            FadeOut(block_M),
            FadeOut(counter_label),
            FadeOut(table),
            ReplacementTransform(counter_value, pi_text)
        )
        
        self.wait(3)
