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
        # Initial Setup
        title = "Calculating Expected Information Gain (EIG)"
        lines = [
            "We calculate expected information using the entropy formula.",
            "Our goal is to maximize the average bits gained.",
            "The word CRANE offers high mathematical surprise.",
            "Bit-Bot computes the average utility of this guess.",
            "A high bit score indicates an excellent opening word."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Entropy formula setup (using Text for stability)
        entropy_math = Text("H = Σ p * log2( 1 / p )", font_size=32, color=WHITE)
        # Issue 37: Placing in area C1-C6
        self.place_in_area(entropy_math, 'C1', 'C6', scale_factor=0.8)
        
        self.play(
            Write(entropy_math),
            self.lecture[0].animate.set_color(BLUE)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Bit-Bot Setup [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg]
        bit_bot = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/robot.svg")
        bit_bot.set_color("#00FF00")
        # Starting position off-screen right
        self.place_at_grid(bit_bot, 'E6', scale_factor=0.8)
        bit_bot.shift(RIGHT * 2)
        
        # Move Bit-Bot to grid D5
        target_pos = self.grid['D5']
        
        self.play(
            bit_bot.animate.move_to(target_pos),
            self.lecture[1].animate.set_color(GREEN)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # CRANE word tiles
        letters = ["C", "R", "A", "N", "E"]
        colors = [GRAY, GRAY, YELLOW, GRAY, GREEN]
        crane_word = VGroup()
        for char, col in zip(letters, colors):
            box = Square(side_length=0.8, fill_color=col, fill_opacity=0.8, stroke_color=WHITE)
            txt = Text(char, font_size=30, color=WHITE)
            txt.move_to(box.get_center())
            crane_word.add(VGroup(box, txt))
        
        crane_word.arrange(RIGHT, buff=0.1)
        # Issue 38: Place in area A2-A5
        self.place_in_area(crane_word, 'A2', 'A5', scale_factor=0.7)

        # Floating probabilities (Issue 39 context: labels between word and formula)
        probs_texts = ["10%", "5%", "20%", "3%", "62%"]
        split_label_group = VGroup(*[Text(p, font_size=20, color=YELLOW) for p in probs_texts])
        split_label_group.arrange(RIGHT, buff=0.4)
        # Issue 39: Place in area B2-B5
        self.place_in_area(split_label_group, 'B2', 'B5', scale_factor=0.6)

        self.play(
            FadeIn(crane_word, shift=DOWN),
            FadeIn(split_label_group, shift=UP),
            self.lecture[2].animate.set_color(YELLOW)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Bit-Bot pulses yellow (#C9B458)
        self.play(
            bit_bot.animate.set_color("#C9B458").scale(1.2),
            self.lecture[3].animate.set_color("#C9B458"),
            run_time=0.6
        )
        self.play(
            bit_bot.animate.set_color("#00FF00").scale(1/1.2),
            run_time=0.6
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Final bit score
        final_value = Text("5.87 Bits", font_size=40, color=WHITE)
        # Place in area E2-E4
        self.place_in_area(final_value, 'E2', 'E4', scale_factor=1.2)
        
        # Glow effect (simulated with a larger, fainter background text)
        glow = final_value.copy().set_style(stroke_width=10, stroke_color=WHITE).set_opacity(0.3)

        self.play(
            FadeIn(glow),
            Write(final_value),
            self.lecture[4].animate.set_color(WHITE)
        )
        self.wait(3)
