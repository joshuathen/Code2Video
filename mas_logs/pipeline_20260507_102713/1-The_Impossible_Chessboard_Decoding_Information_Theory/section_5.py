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

class Section5Scene(TeachingScene):
    def construct(self):
        # Setup layout
        title = "Information Theory: From Puzzles to Data"
        lines = [
            'This logic powers Hamming Codes used in modern computing.',
            'It detects and fixes errors in digital data transmissions.',
            'From space photos to SSDs, XOR ensures data integrity.'
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_BINARY = "#FFFFFF"
        COLOR_ERROR = "#FF0000"
        COLOR_FIX = "#00FF00"
        COLOR_SAT = "#1E90FF"
        COLOR_CHIP = "#FFD700"
        COLOR_SSD = "#FFFFFF"
        COLOR_HIGHLIGHT = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_HIGHLIGHT)
        
        # Create a stream of binary data
        binary_bits = ["1", "0", "1", "1", "0", "0", "1", "0"]
        bits_group = VGroup(*[Text(b, font="Monospace", font_size=36, color=COLOR_BINARY) for b in binary_bits])
        bits_group.arrange(RIGHT, buff=0.4)
        self.place_in_area(bits_group, "B1", "B6")
        
        # Label segment as Hamming Code
        hamming_rect = SurroundingRectangle(bits_group, color=COLOR_BINARY, buff=0.2)
        hamming_label = Text("Hamming Code", font_size=20, color=COLOR_BINARY)
        self.place_at_grid(hamming_label, "C3", scale_factor=0.8)
        hamming_label.next_to(hamming_rect, DOWN)

        self.play(Write(bits_group))
        self.play(Create(hamming_rect), FadeIn(hamming_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(COLOR_HIGHLIGHT)

        # One bit turns red (error)
        error_bit_index = 4
        error_bit = bits_group[error_bit_index]
        
        self.play(error_bit.animate.set_color(COLOR_ERROR))
        self.wait(0.5)
        
        # Checker icon flips it back
        checker = VGroup(
            Square(side_length=0.5, color=COLOR_FIX),
            Line(start=0.2*LEFT+0.1*DOWN, end=ORIGIN, color=COLOR_FIX),
            Line(start=ORIGIN, end=0.2*RIGHT+0.3*UP, color=COLOR_FIX)
        )
        self.place_at_grid(checker, "A5", scale_factor=0.8)
        checker.next_to(error_bit, UP, buff=0.3)
        
        self.play(FadeIn(checker, shift=DOWN))
        self.play(error_bit.animate.set_color(COLOR_FIX))
        self.wait(0.5)
        self.play(FadeOut(checker))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(COLOR_HIGHLIGHT)

        # Clear previous elements
        self.play(FadeOut(bits_group, hamming_rect, hamming_label))

        # XOR Symbol
        xor_circle = Circle(radius=0.6, color=WHITE)
        xor_plus = VGroup(
            Line(0.4*LEFT, 0.4*RIGHT, color=WHITE),
            Line(0.4*UP, 0.4*DOWN, color=WHITE)
        )
        xor_symbol = VGroup(xor_circle, xor_plus)
        self.place_in_area(xor_symbol, "C3", "D4", scale_factor=1.0)
        xor_text = Text("XOR", font_size=24).next_to(xor_symbol, DOWN)

        # Satellite Icon
        sat_body = Rectangle(width=0.6, height=0.4, color=COLOR_SAT)
        sat_wing_l = Rectangle(width=0.4, height=0.2, color=COLOR_SAT).next_to(sat_body, LEFT, buff=0.1)
        sat_wing_r = Rectangle(width=0.4, height=0.2, color=COLOR_SAT).next_to(sat_body, RIGHT, buff=0.1)
        satellite = VGroup(sat_body, sat_wing_l, sat_wing_r)
        self.place_at_grid(satellite, "B2", scale_factor=0.8)
        sat_label = Text("Satellite", font_size=16, color=COLOR_SAT).next_to(satellite, UP)

        # Chip Icon
        chip_base = Square(side_length=0.6, color=COLOR_CHIP)
        # Fixed the Line initialization by providing start and end instead of an invalid 'length' argument
        pins = VGroup(*[Line(start=ORIGIN, end=UP * 0.15, color=COLOR_CHIP) for _ in range(8)])
        for i, pin in enumerate(pins[:2]): pin.next_to(chip_base, UP, buff=0).shift((i-0.5)*0.3*RIGHT)
        for i, pin in enumerate(pins[2:4]): pin.next_to(chip_base, DOWN, buff=0).shift((i-0.5)*0.3*RIGHT)
        for i, pin in enumerate(pins[4:6]): pin.next_to(chip_base, LEFT, buff=0).shift((i-0.5)*0.3*UP)
        for i, pin in enumerate(pins[6:8]): pin.next_to(chip_base, RIGHT, buff=0).shift((i-0.5)*0.3*UP)
        chip = VGroup(chip_base, pins)
        self.place_at_grid(chip, "B5", scale_factor=0.8)
        chip_label = Text("Chip", font_size=16, color=COLOR_CHIP).next_to(chip, UP)

        # SSD Icon
        ssd_case = Rectangle(width=0.5, height=0.7, color=COLOR_SSD)
        ssd_lines = VGroup(*[Line(0.15*LEFT, 0.15*RIGHT, color=COLOR_SSD) for _ in range(3)]).arrange(DOWN, buff=0.1).move_to(ssd_case)
        ssd = VGroup(ssd_case, ssd_lines)
        self.place_at_grid(ssd, "E3", scale_factor=0.8)
        ssd_label = Text("SSD", font_size=16, color=COLOR_SSD).next_to(ssd, DOWN)

        self.play(Create(xor_symbol), Write(xor_text))
        self.play(FadeIn(satellite, sat_label), FadeIn(chip, chip_label), FadeIn(ssd, ssd_label))
        
        # Connect icons to XOR
        lines_to_xor = VGroup(
            Line(satellite.get_bottom(), xor_symbol.get_top(), color=COLOR_SAT, stroke_width=2),
            Line(chip.get_bottom(), xor_symbol.get_top(), color=COLOR_CHIP, stroke_width=2),
            Line(ssd.get_top(), xor_symbol.get_bottom(), color=COLOR_SSD, stroke_width=2)
        )
        self.play(Create(lines_to_xor))
        self.wait(2)

        # Final Cleanup
        self.lecture[2].set_color(WHITE)
        self.wait(1)
