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
        # Section data
        title = "Prerequisite: The Parity Bit (Detection)"
        lines = [
            "We can add a parity bit for simple detection.",
            "It ensures the total count of ones is even.",
            "If an error occurs, the count becomes odd."
        ]
        self.setup_layout(title, lines)
        
        # === Animation for Lecture Line 1 ===
        # The first line is active (color: White - matching the white bits)
        # Display data bits [1, 0, 1] in white boxes
        bits_vals = ["1", "0", "1"]
        boxes = VGroup(*[Square(side_length=0.8, color=WHITE) for _ in range(4)])
        bit_texts = VGroup(*[Text(val, font_size=36, color=WHITE) for val in bits_vals])
        
        # Position boxes in grid row B: B2, B3, B4, B5
        for i in range(4):
            self.place_at_grid(boxes[i], f"B{i+2}")
            
        for i in range(3):
            bit_texts[i].move_to(boxes[i].get_center())
            
        self.play(Create(boxes), Write(bit_texts))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line with light blue (#ADD8E6) to match the parity bit
        self.play(
            self.lecture[1].animate.set_color("#ADD8E6")
        )
        
        # Parity bit: '0' (Even Parity) in light blue (#ADD8E6)
        parity_bit = Text("0", font_size=36, color="#ADD8E6")
        parity_bit.move_to(boxes[3].get_center())
        
        label_parity = Text("Parity Bit", font_size=20, color="#ADD8E6")
        # Position label across grid units for centering (A4-A6) - Resolved Issue 26
        self.place_in_area(label_parity, 'A4', 'A6', scale_factor=0.6)
        
        self.play(Write(parity_bit), FadeIn(label_parity))
        self.wait(2)
        
        # === Animation for Lecture Line 3 ===
        # Highlight third line with red (#FF0000) to match the error state
        self.play(
            self.lecture[1].animate.set_color(WHITE), # Reset previous highlight
            self.lecture[2].animate.set_color("#FF0000")
        )
        
        # Change second bit (index 1) from '0' to '1' in red (#FF0000)
        error_bit = Text("1", font_size=36, color="#FF0000")
        error_bit.move_to(boxes[1].get_center())
        
        # Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/alarm.svg]
        # Resolve Issue 22 (Integration) and Issue 27 (Positioning)
        alarm_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/alarm.svg")
        alarm_asset.set_color("#FF0000")
        self.place_at_grid(alarm_asset, 'E3', scale_factor=1.2)
        
        self.play(
            FadeOut(bit_texts[1]),
            FadeIn(error_bit),
            boxes[1].animate.set_color("#FF0000")
        )
        self.play(Flash(alarm_asset, color="#FF0000", line_length=0.4), FadeIn(alarm_asset))
        self.wait(2)
