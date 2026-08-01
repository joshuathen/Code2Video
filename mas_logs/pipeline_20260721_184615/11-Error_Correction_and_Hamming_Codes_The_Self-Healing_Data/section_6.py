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
        # Setup title and lecture lines
        # Using the 5-line script consistent with the storyboard animation steps and existing code
        title_text = "Correction: Finding the 'Syndrome'"
        lecture_lines = [
            "To fix errors, we re-calculate all parity bits.",
            "Failing checks indicate something changed during transmission.",
            "Add the position numbers of all failing parity bits.",
            "This sum identifies the exact index of the error.",
            "Simply flip that bit back to restore the data."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Hex Colors (L008)
        HIGHLIGHT = "#FFFF00"
        ERROR = "#FF0000"
        CORRECT = "#00FF00"
        WHITE_COLOR = "#FFFFFF"
        BLUE_ACCENT = "#00FFFF"
        GRAY_TEXT = "#808080"

        # Define the bit sequence (Hamming 7,4)
        # Positions: 1, 2, 3, 4, 5, 6, 7
        # Error at position 3 (D3)
        positions = [1, 2, 3, 4, 5, 6, 7]
        bit_labels = ["P1", "P2", "D3", "P4", "D5", "D6", "D7"]
        bit_values = [0, 0, 1, 0, 0, 0, 0]
        
        bits_group = VGroup()
        for i in range(7):
            # Proximity Rule (L002): labels within 1 unit
            square = Square(side_length=0.7, color=WHITE_COLOR)
            val = Text(str(bit_values[i]), font_size=24, color=WHITE_COLOR)
            pos_label = Text(str(positions[i]), font_size=16, color=BLUE_ACCENT).next_to(square, UP, buff=0.1)
            name_label = Text(bit_labels[i], font_size=16, color=GRAY_TEXT).next_to(square, DOWN, buff=0.1)
            bit_mobject = VGroup(square, val, pos_label, name_label)
            bits_group.add(bit_mobject)
        
        bits_group.arrange(RIGHT, buff=0.2)
        # Issue 43: scale_factor=0.8
        self.place_in_area(bits_group, "B2", "C6", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # "To fix errors, we re-calculate all parity bits."
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT))
        
        # Storyboard: Highlight recalculation of P1, P2, P4
        self.play(FadeIn(bits_group))
        self.play(
            Indicate(bits_group[0][0], color=WHITE_COLOR), # L004
            Indicate(bits_group[1][0], color=WHITE_COLOR),
            Indicate(bits_group[3][0], color=WHITE_COLOR),
        )
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # "Failing checks indicate something changed during transmission."
        self.play(
            self.lecture[0].animate.set_color(WHITE_COLOR),
            self.lecture[1].animate.set_color(HIGHLIGHT)
        )
        # Storyboard: P1 and P2 turn red (#FF0000)
        self.play(
            bits_group[0][0].animate.set_color(ERROR),
            bits_group[1][0].animate.set_color(ERROR),
            bits_group[0][1].animate.set_color(ERROR),
            bits_group[1][1].animate.set_color(ERROR),
        )
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # "Add the position numbers of all failing parity bits."
        self.play(
            self.lecture[1].animate.set_color(WHITE_COLOR),
            self.lecture[2].animate.set_color(HIGHLIGHT)
        )
        
        # Storyboard: Display the math "1 + 2 = 3"
        # Issue 41: math_sum positioned at D4
        math_result = Text("1 + 2 = 3", font_size=36, color=WHITE_COLOR)
        self.place_at_grid(math_result, 'D4', scale_factor=1.0)
        
        self.play(Write(math_result))
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        # "This sum identifies the exact index of the error."
        self.play(
            self.lecture[2].animate.set_color(WHITE_COLOR),
            self.lecture[3].animate.set_color(HIGHLIGHT)
        )
        # Storyboard: The bit at position 3 (D3) is highlighted with a yellow box (#FFFF00).
        highlight_box = SurroundingRectangle(bits_group[2], color=HIGHLIGHT, buff=0.1)
        self.play(Create(highlight_box))
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # "Simply flip that bit back to restore the data."
        self.play(
            self.lecture[3].animate.set_color(WHITE_COLOR),
            self.lecture[4].animate.set_color(HIGHLIGHT)
        )
        
        # Storyboard: Bit D3 flashes green (#00FF00) and flips its value.
        # Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/based.svg]
        
        # Flash green
        self.play(Flash(bits_group[2][0], color=CORRECT, flash_radius=0.5))
        
        # Flip value
        corrected_val = Text("0", font_size=24, color=CORRECT).move_to(bits_group[2][1])
        
        # Load and place asset (Issue 27)
        based_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/based.svg")
        # Position it within 1 grid unit of target object
        based_icon.scale(0.3).next_to(bits_group[2], RIGHT, buff=0.2)
        
        self.play(
            Transform(bits_group[2][1], corrected_val),
            bits_group[2][0].animate.set_color(CORRECT),
            FadeIn(based_icon),
            Uncreate(highlight_box)
        )
        self.wait(2.0)
