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
        # Data from storyboard
        title = "Prerequisite: Binary Counting as a Pulse"
        lines = [
            "Binary counting uses a sequence of flipping bits.",
            "Each bit position represents a power of two.",
            "Observe which bit flips during each increment."
        ]
        
        self.setup_layout(title, lines)
        
        # Colors for alignment
        COLOR_LINE1 = "#FFFF00" # Yellow
        COLOR_LINE2 = "#00FF00" # Green
        COLOR_LINE3 = "#00FFFF" # Cyan
        
        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(COLOR_LINE1))
        
        # Initial bits "000" in White
        bit3 = Text("0", font_size=60, color=WHITE)
        bit2 = Text("0", font_size=60, color=WHITE)
        bit1 = Text("0", font_size=60, color=WHITE)
        
        bits_group = VGroup(bit3, bit2, bit1).arrange(RIGHT, buff=0.5)
        # Issue 37: Apply scale_factor=0.8 to bits_group
        self.place_in_area(bits_group, "B2", "C5", scale_factor=0.8)
        
        self.play(Write(bits_group))
        self.wait(0.5)
        
        # Transition 000 -> 001, rightmost bit yellow
        new_bit1_y = Text("1", font_size=60, color=COLOR_LINE1).move_to(bit1)
        
        self.play(Transform(bit1, new_bit1_y))
        self.wait(1)
        
        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_LINE2)
        )
        
        # Labels for powers of 2 (4, 2, 1) in Green
        p3 = Text("2^2", font_size=24, color=COLOR_LINE2).next_to(bit3, UP, buff=0.4)
        p2 = Text("2^1", font_size=24, color=COLOR_LINE2).next_to(bit2, UP, buff=0.4)
        p1 = Text("2^0", font_size=24, color=COLOR_LINE2).next_to(bit1, UP, buff=0.4)
        
        self.play(FadeIn(p3), FadeIn(p2), FadeIn(p1))
        
        # Transition 001 -> 010, highlighting bits that change in green
        # Bits 1 and 2 flip
        new_bit2_g = Text("1", font_size=60, color=COLOR_LINE2).move_to(bit2)
        new_bit1_g = Text("0", font_size=60, color=COLOR_LINE2).move_to(bit1)
        
        self.play(
            Transform(bit2, new_bit2_g),
            Transform(bit1, new_bit1_g)
        )
        self.wait(0.5)
        
        # Transition 010 -> 011, Bit 1 flips
        new_bit1_g2 = Text("1", font_size=60, color=COLOR_LINE2).move_to(bit1)
        self.play(Transform(bit1, new_bit1_g2))
        self.wait(1)

        # Reset bit colors to White for the final demonstration
        self.play(
            bit1.animate.set_color(WHITE),
            bit2.animate.set_color(WHITE),
            bit3.animate.set_color(WHITE)
        )

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_LINE3)
        )
        
        # Asset: lightbulb.svg
        # Issue 28: Integrated SVGMobject asset
        bulb_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/lightbulb.svg"
        
        bulb3 = SVGMobject(bulb_path).set_color(WHITE).set_stroke(width=1).set_fill(WHITE, opacity=0.2)
        bulb2 = SVGMobject(bulb_path).set_color(WHITE).set_stroke(width=1).set_fill(WHITE, opacity=1.0)
        bulb1 = SVGMobject(bulb_path).set_color(WHITE).set_stroke(width=1).set_fill(WHITE, opacity=1.0)
        
        bulbs_group = VGroup(bulb3, bulb2, bulb1).arrange(RIGHT, buff=0.8)
        # Issue 35 & 36: Position bulbs_group in D2-E5 to fix gap and vertical positioning
        self.place_in_area(bulbs_group, "D2", "E5", scale_factor=0.6)
        
        # Bit labels below bulbs
        l3 = Text("Bit 3", font_size=20, color=WHITE).next_to(bulb3, DOWN, buff=0.2)
        l2 = Text("Bit 2", font_size=20, color=WHITE).next_to(bulb2, DOWN, buff=0.2)
        l1 = Text("Bit 1", font_size=20, color=WHITE).next_to(bulb1, DOWN, buff=0.2)
        
        self.play(FadeIn(bulbs_group), FadeIn(l3), FadeIn(l2), FadeIn(l1))
        
        # Pulsing Sequence: 011(3) -> 100(4) -> 101(5) -> 110(6) -> 111(7) -> 000(0)
        curr_val = 3
        for _ in range(5):
            curr_val = (curr_val + 1) % 8
            b_str = format(curr_val, '03b')
            
            # Prepare next bit texts
            next_t3 = Text(b_str[0], font_size=60, color=WHITE).move_to(bit3)
            next_t2 = Text(b_str[1], font_size=60, color=WHITE).move_to(bit2)
            next_t1 = Text(b_str[2], font_size=60, color=WHITE).move_to(bit1)
            
            # Bulb appearance: Opacity 1.0 for ON, 0.2 for OFF
            op3 = 1.0 if b_str[0] == '1' else 0.2
            op2 = 1.0 if b_str[1] == '1' else 0.2
            op1 = 1.0 if b_str[2] == '1' else 0.2
            
            self.play(
                Transform(bit3, next_t3),
                Transform(bit2, next_t2),
                Transform(bit1, next_t1),
                bulb3.animate.set_fill(WHITE, opacity=op3),
                bulb2.animate.set_fill(WHITE, opacity=op2),
                bulb1.animate.set_fill(WHITE, opacity=op1),
                run_time=0.6
            )
            self.wait(0.1)

        self.wait(2)
