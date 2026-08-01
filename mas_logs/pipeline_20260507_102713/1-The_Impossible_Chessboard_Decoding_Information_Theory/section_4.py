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
        # Initialize layout
        lines = [
            'Alice needs the signature to match the target square.',
            'She XORs the current signature with the target index.',
            'The result tells Alice exactly which coin to flip.',
            'Flipping that coin adjusts the signature to the target.',
            'Bob calculates the new XOR sum to find it.'
        ]
        self.setup_layout("The Strategy: The 'Magic' Flip", lines)

        # Colors
        COLOR_SIGNATURE = "#1E90FF"  # Dodger Blue
        COLOR_TARGET = "#FF0000"     # Red
        COLOR_FLIP = "#00FF00"       # Green

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        sig_label = Text("Current Signature: 10", font_size=20, color=COLOR_SIGNATURE)
        target_label = Text("Target Square: 1", font_size=20, color=COLOR_TARGET)
        
        self.place_at_grid(sig_label, "A2", scale_factor=1.0)
        self.place_at_grid(target_label, "A5", scale_factor=1.0)
        
        self.play(Write(sig_label), Write(target_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Binary representation: 10 (1010) and 1 (0001)
        # Using VGroups to manage bits individually
        bits_sig = VGroup(*[Text(b, font_size=30, color=COLOR_SIGNATURE) for b in "1010"]).arrange(RIGHT, buff=0.4)
        bits_target = VGroup(*[Text(b, font_size=30, color=COLOR_TARGET) for b in "0001"]).arrange(RIGHT, buff=0.4)
        xor_symbol = Text("XOR", font_size=24, color=WHITE)
        line_divider = Line(LEFT, RIGHT, color=WHITE).scale(1.5)
        
        xor_math_group = VGroup(bits_sig, xor_symbol, bits_target, line_divider).arrange(DOWN, buff=0.3)
        self.place_in_area(xor_math_group, "B2", "C5", scale_factor=1.0)
        
        self.play(FadeIn(xor_math_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Bit by bit calculation
        res_bits = "1011" # 10 XOR 1 = 11
        bits_result = VGroup(*[Text(b, font_size=30, color=COLOR_FLIP) for b in res_bits]).arrange(RIGHT, buff=0.4)
        bits_result.next_to(line_divider, DOWN, buff=0.3)
        
        flip_label = Text("Flip Index: 11", font_size=20, color=COLOR_FLIP).next_to(bits_result, RIGHT, buff=0.5)
        
        for i in range(4):
            self.play(Indicate(bits_sig[i]), Indicate(bits_target[i]), run_time=0.5)
            self.play(FadeIn(bits_result[i]), run_time=0.3)
            
        self.play(Write(flip_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        
        # Create a 4x4 Grid for representation
        grid_squares = VGroup()
        for i in range(16):
            sq = Square(side_length=0.6, stroke_width=2, color=WHITE)
            label = Text(str(i), font_size=14, color=GRAY)
            sq_group = VGroup(sq, label)
            grid_squares.add(sq_group)
        
        grid_squares.arrange_in_grid(rows=4, cols=4, buff=0.1)
        self.place_in_area(grid_squares, "D2", "F5", scale_factor=0.9)
        
        # Identify specific indices
        idx_sig = 10
        idx_target = 1
        idx_flip = 11
        
        grid_squares[idx_sig][0].set_stroke(COLOR_SIGNATURE, 4)
        grid_squares[idx_target][0].set_stroke(COLOR_TARGET, 4)
        
        self.play(Create(grid_squares))
        
        # Flip index 11
        arrow = Arrow(start=UP, end=DOWN, color=COLOR_FLIP).scale(0.5)
        arrow.next_to(grid_squares[idx_flip], UP, buff=0.1)
        
        self.play(FadeIn(arrow))
        self.play(
            grid_squares[idx_flip][0].animate.set_fill(COLOR_FLIP, opacity=0.5),
            Flash(grid_squares[idx_flip], color=COLOR_FLIP)
        )
        
        # Update labels to show change
        new_sig_label = Text("New Signature: 1", font_size=20, color=COLOR_TARGET)
        self.place_at_grid(new_sig_label, "A2", scale_factor=1.0)
        
        self.play(
            FadeOut(sig_label),
            FadeIn(new_sig_label),
            FadeOut(arrow)
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        bob_label = Text("Bob", font_size=24, color=WHITE).next_to(grid_squares, RIGHT, buff=0.5)
        
        # Visual pulse of the target square
        self.play(Write(bob_label))
        self.play(
            grid_squares[idx_target][0].animate.set_fill(COLOR_TARGET, opacity=0.8),
            run_time=0.5
        )
        self.play(
            grid_squares[idx_target][0].animate.set_fill(COLOR_TARGET, opacity=0.2),
            run_time=0.5
        )
        
        success_msg = Text("Match Found!", font_size=24, color=COLOR_TARGET)
        self.place_at_grid(success_msg, "B6", scale_factor=1.0)
        self.play(Write(success_msg))
        
        self.wait(2)

# Issue Resolution:
# Resolving Issue 12: Generated complete Manim code for Section 4 with precise grid positioning,
# logic-aligned XOR calculation, and visual representation of Alice's strategy and Bob's verification.

# update_issue(12, under_review=True, resolution_note='Section 4 Manim code implemented with XOR calculation and grid-based strategy visualization.')
