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
        lines = [
            "Hamming codes enable self-healing data in modern systems.",
            "ECC memory uses these codes to prevent server crashes.",
            "Error correction ensures reliable computing in a noisy world."
        ]
        self.setup_layout("Real-World Application & Summary", lines)
        
        # Colors for highlights
        color_1 = BLUE_B
        color_2 = YELLOW_B
        color_3 = GREEN_B

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_1))
        
        # Create a basic "Server Rack"
        rack_frame = Rectangle(width=4, height=5, color=GREY_D)
        rack_shelves = VGroup(*[Line(rack_frame.get_left(), rack_frame.get_right(), color=GREY_D) for _ in range(5)])
        rack_shelves.arrange(DOWN, buff=0.8)
        rack_dots = VGroup(*[Dot(color=GREEN_E, radius=0.05) for _ in range(15)])
        rack_dots.arrange_in_grid(5, 3, buff=0.4)
        server_rack = VGroup(rack_frame, rack_shelves, rack_dots)
        
        self.place_in_area(server_rack, "A2", "F5", scale_factor=0.6)
        self.play(FadeIn(server_rack))
        self.wait(1)

        # Create ECC RAM module
        ram_pcb = Rectangle(width=4, height=1.5, color=GREEN_E, fill_opacity=1)
        chips = VGroup(*[Square(side_length=0.4, color=BLACK, fill_opacity=1) for _ in range(8)])
        chips.arrange(RIGHT, buff=0.1)
        labels_ram = Text("ECC RAM", font_size=20, color=WHITE)
        ram_module = VGroup(ram_pcb, chips, labels_ram)
        
        # Zoom into RAM (Simulated by fading rack out and RAM in larger)
        # Fix for Issue 40: Position and scale RAM module to avoid background overlap
        self.play(
            FadeOut(server_rack),
            FadeIn(self.place_in_area(ram_module, "C2", "E5", scale_factor=0.7))
        )
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_2)
        )
        
        # Hide RAM, show bit sequence
        self.play(FadeOut(ram_module))
        
        bit_values = [1, 0, 1, 1, 0, 0, 1]
        bits = VGroup(*[
            VGroup(
                Square(side_length=0.7, color=WHITE),
                Text(str(b), font_size=24)
            ) for b in bit_values
        ]).arrange(RIGHT, buff=0.2)
        
        self.place_in_area(bits, "C1", "C6", scale_factor=0.8)
        self.play(FadeIn(bits))
        self.wait(1)
        
        # Error occurs (flip 4th bit: index 3)
        error_bit_square = bits[3][0]
        error_bit_text = bits[3][1]
        
        self.play(
            error_bit_square.animate.set_color(RED),
            error_bit_text.animate.set_color(RED)
        )
        
        flipped_text = Text("0", font_size=24, color=RED)
        flipped_text.move_to(error_bit_text.get_center())
        self.play(Transform(error_bit_text, flipped_text))
        
        err_label = Text("Error!", color=RED, font_size=18)
        self.place_at_grid(err_label, "B4", scale_factor=1.0)
        self.play(Write(err_label))
        self.wait(1)
        
        # Correction
        hamming_label = Text("Hamming Code Correcting...", color=GREEN, font_size=20)
        # Fix for Issue 38: Use place_in_area for multi-word label
        self.place_in_area(hamming_label, "D1", "D6", scale_factor=0.8)
        self.play(Write(hamming_label))
        
        corrected_text = Text("1", font_size=24, color=GREEN)
        corrected_text.move_to(error_bit_text.get_center())
        
        self.play(
            Transform(error_bit_text, corrected_text),
            error_bit_square.animate.set_color(GREEN),
            FadeOut(err_label)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_3)
        )
        
        # Fade everything and show final summary text
        self.play(
            FadeOut(bits),
            FadeOut(hamming_label)
        )
        
        summary_text = Text("Hamming Codes:\nSelf-Healing Data", color="#00FF00", font_size=32, t2c={"Self-Healing Data": "#00FF00"})
        # Fix for Issue 39: Balance summary text composition
        self.place_in_area(summary_text, "B2", "E5", scale_factor=0.8)
        
        self.play(Write(summary_text))
        self.wait(3)
