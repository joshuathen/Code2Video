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
        # Title and Lecture Lines
        lecture_lines = [
            'A parity bit ensures the count of ones is even.',
            'One bit flip makes the total count odd.',
            'Simple parity detects errors but cannot locate them.'
        ]
        self.setup_layout("Prerequisite: The Simple Parity Bit", lecture_lines)
        
        # Colors
        COLOR_WHITE = "#FFFFFF"
        COLOR_GREEN = "#00FF00"
        COLOR_RED = "#FF0000"

        # === Animation for Lecture Line 1 ===
        # Display '110' (#FFFFFF) and append even parity bit '0' (#00FF00).
        self.lecture[0].set_color(COLOR_GREEN)
        
        data_bits = VGroup(
            Text("1", color=COLOR_WHITE),
            Text("1", color=COLOR_WHITE),
            Text("0", color=COLOR_WHITE)
        ).arrange(RIGHT, buff=0.3)
        self.place_in_area(data_bits, "B2", "B4", scale_factor=1.2)
        
        parity_bit = Text("0", color=COLOR_GREEN)
        self.place_at_grid(parity_bit, "B5", scale_factor=1.2)
        
        data_label = Text("Data Bits", font_size=18, color=COLOR_WHITE)
        self.place_at_grid(data_label, "A3", scale_factor=0.8)
        
        parity_label = Text("Even Parity", font_size=18, color=COLOR_GREEN)
        self.place_at_grid(parity_label, "A5", scale_factor=0.8)
        
        self.play(FadeIn(data_bits), FadeIn(data_label))
        self.wait(0.5)
        self.play(Write(parity_bit), FadeIn(parity_label))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Flip the last bit to '1' (#FF0000); show total 1s is 3 (Odd).
        self.lecture[0].set_color(COLOR_WHITE)
        self.lecture[1].set_color(COLOR_RED)
        
        error_bit = Text("1", color=COLOR_RED)
        self.place_at_grid(error_bit, "B5", scale_factor=1.2)
        
        # Note: We replace the parity bit with an error bit to simulate corruption
        count_msg = Text("Total 1s = 3 (Odd!)", color=COLOR_RED, font_size=24)
        # Issue 33: Expand to C2-C6 for horizontal room
        self.place_in_area(count_msg, "C2", "C6", scale_factor=1.0)
        
        self.play(Transform(parity_bit, error_bit))
        self.play(Write(count_msg))
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Display 'Error Detected' (#FF0000) near Bob [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/bob.svg]; location unknown.
        self.lecture[1].set_color(COLOR_WHITE)
        self.lecture[2].set_color(COLOR_RED)
        
        error_alert = Text("Error Detected", color=COLOR_RED, font_size=30)
        # Issue 31: Restrict to D2-D5 to avoid vertical crowding
        self.place_in_area(error_alert, "D2", "D5", scale_factor=1.0)
        
        # Issue 26: Use Bob SVG Asset
        bob_icon = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/bob.svg")
        bob_icon.set_color(COLOR_WHITE)
        bob_label = Text("Bob", color=COLOR_WHITE, font_size=16).next_to(bob_icon, DOWN, buff=0.1)
        bob_vgroup = VGroup(bob_icon, bob_label)
        self.place_at_grid(bob_vgroup, "D6", scale_factor=0.8)
        
        unknown_label = Text("But which bit is wrong?", color=COLOR_RED, font_size=20)
        # Issue 32: Position in E2-E5, scale 0.8
        self.place_in_area(unknown_label, "E2", "E5", scale_factor=0.8)
        
        self.play(FadeIn(bob_vgroup))
        self.play(FadeIn(error_alert))
        self.wait(0.5)
        self.play(Write(unknown_label))
        self.wait(3)
