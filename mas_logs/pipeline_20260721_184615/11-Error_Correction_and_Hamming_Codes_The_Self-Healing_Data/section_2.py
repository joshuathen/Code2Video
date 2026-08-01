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
        # Title and Lecture Lines for Section 2
        title_text = "Prerequisite: The Simple Parity Bit"
        lecture_lines = [
            "Parity bits detect if an error occurred.",
            "Even parity makes the total count of ones even.",
            "However, it cannot locate or fix the error."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show binary "101" (#FFFFFF); a green '0' (#00FF00) appears at the end.
        self.lecture[0].set_color("#FFFF00")
        
        bit1 = Text("1", font_size=48, color="#FFFFFF")
        bit2 = Text("0", font_size=48, color="#FFFFFF")
        bit3 = Text("1", font_size=48, color="#FFFFFF")
        
        # Fix Issue 33: Shift bits left
        self.place_at_grid(bit1, "C1")
        self.place_at_grid(bit2, "C2")
        self.place_at_grid(bit3, "C3")
        
        data_bits = VGroup(bit1, bit2, bit3)
        
        # Fix Issue 33: Shift parity bit left
        parity_bit = Text("0", font_size=48, color="#00FF00")
        self.place_at_grid(parity_bit, "C4")
        
        # Fix Issue 33: Shift parity label left
        parity_label = Text("Parity Bit", font_size=24, color="#00FF00")
        self.place_at_grid(parity_label, "B4", scale_factor=0.7)
        
        self.play(Write(data_bits))
        self.play(FadeIn(parity_bit), FadeIn(parity_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Sum the '1's; a box highlights the two '1's and parity '0'.
        self.lecture[0].set_color("#FFFFFF")
        self.lecture[1].set_color("#FFFF00")
        
        # Box highlighting the '1's and the parity '0'
        all_bits = VGroup(bit1, bit2, bit3, parity_bit)
        highlight_box = SurroundingRectangle(all_bits, color="#FFFF00", buff=0.15)
        
        # Fix Issue 34: Align sum formula
        sum_formula = Text("1 + 0 + 1 + 0 = 2", font_size=32, color="#FFFF00")
        self.place_in_area(sum_formula, "D1", "D4", scale_factor=0.8)
        
        # Safety for even_label
        even_label = Text("(Even)", font_size=28, color="#00FF00")
        self.place_in_area(even_label, "D5", "D6", scale_factor=0.7)
        
        self.play(Create(highlight_box))
        self.play(Write(sum_formula))
        self.play(FadeIn(even_label))
        self.play(Indicate(highlight_box))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Flip a bit to '1'; the entire block turns red (#FF0000) indicating error.
        self.lecture[1].set_color("#FFFFFF")
        self.lecture[2].set_color("#FFFF00")
        
        # Fix Issue 34: Align error bit
        error_bit = Text("1", font_size=48, color="#FF0000")
        self.place_at_grid(error_bit, "C2")
        
        # Transition to error state
        error_box = SurroundingRectangle(VGroup(bit1, error_bit, bit3, parity_bit), color="#FF0000", buff=0.2)
        
        # Fix Issue 34: Align error sum
        error_sum = Text("1 + 1 + 1 + 0 = 3", font_size=32, color="#FF0000")
        self.place_in_area(error_sum, "D1", "D4", scale_factor=0.8)
        
        # Fix Issue 32: Use place_in_area for odd_label
        odd_label = Text("(Odd - Error!)", font_size=28, color="#FF0000")
        self.place_in_area(odd_label, "D5", "D6", scale_factor=0.7)
        
        self.play(
            FadeOut(highlight_box),
            FadeOut(sum_formula),
            FadeOut(even_label),
            ReplacementTransform(bit2, error_bit),
            bit1.animate.set_color("#FF0000"),
            bit3.animate.set_color("#FF0000"),
            parity_bit.animate.set_color("#FF0000"),
            parity_label.animate.set_color("#FF0000")
        )
        
        self.play(
            Create(error_box),
            Write(error_sum),
            FadeIn(odd_label)
        )
        self.play(Indicate(error_box))
        self.wait(2)
