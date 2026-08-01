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
        title_str = "The Magic Trick: Locating and Fixing"
        lecture_lines = [
            "Let's find the liar by checking all circles.",
            "If circles one and two fail, add them.",
            "One plus two equals three: the error's location!",
            "Simply flip bit three to fix the data.",
            "Error correction achieved without a re-send."
        ]
        self.setup_layout(title_str, lecture_lines)

        # Colors
        C_NEUTRAL = WHITE
        C_FAIL = RED_A
        C_PASS = GREEN_A
        C_HIGHLIGHT = YELLOW

        # Bits representation
        # Bit values (Hamming 7,4): 1 1 0 1 1 0 0 (with an error at position 3)
        bits_values = ["1", "1", "0", "1", "1", "0", "0"] 
        bit_boxes = VGroup(*[Square(side_length=0.6, color=C_NEUTRAL) for _ in range(7)])
        bit_labels = VGroup(*[Text(val, font_size=20) for val in bits_values])
        index_labels = VGroup(*[Text(str(i+1), font_size=12, color=GRAY) for i in range(7)])
        
        # Bits group creation
        bits_group = VGroup()
        for i in range(7):
            bit_unit = VGroup(bit_boxes[i], bit_labels[i], index_labels[i])
            bit_labels[i].move_to(bit_boxes[i].get_center())
            index_labels[i].next_to(bit_boxes[i], DOWN, buff=0.1)
            bits_group.add(bit_unit)
        
        bits_group.arrange(RIGHT, buff=0.1)
        
        # Issue 35: The bits_group is scaled too small (0.9) for the D1-D6 area.
        # Fix: Line 79: self.place_in_area(self.bits_group, 'D3', 'D4', scale_factor=1.8)
        self.place_in_area(bits_group, 'D3', 'D4', scale_factor=1.8)

        # Parity Status Indicators
        p1_indicator = VGroup(Circle(radius=0.4, color=C_NEUTRAL), Text("P1", font_size=20))
        p2_indicator = VGroup(Circle(radius=0.4, color=C_NEUTRAL), Text("P2", font_size=20))
        p4_indicator = VGroup(Circle(radius=0.4, color=C_NEUTRAL), Text("P4", font_size=20))
        
        for p in [p1_indicator, p2_indicator, p4_indicator]:
            p[1].move_to(p[0].get_center())

        self.place_at_grid(p1_indicator, "B2")
        self.place_at_grid(p2_indicator, "B3")
        self.place_at_grid(p4_indicator, "B4")

        # Assets
        tower_svg_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/tower.svg"
        checkmark_svg_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/checkmark.svg"

        # === Animation for Lecture Line 1 ===
        # Let's find the liar by checking all circles.
        self.lecture[0].set_color(C_HIGHLIGHT)
        self.play(Create(bits_group))
        self.play(Create(p1_indicator), Create(p2_indicator), Create(p4_indicator))
        
        # Initial fail state for parity checks (1 + 2 = 3 error)
        self.play(
            p1_indicator[0].animate.set_color(C_FAIL),
            p2_indicator[0].animate.set_color(C_FAIL),
            p4_indicator[0].animate.set_color(C_PASS),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # If circles one and two fail, add them.
        self.lecture[0].set_color(C_NEUTRAL)
        self.lecture[1].set_color(C_HIGHLIGHT)
        
        # Issue 25 & 37: Use tower asset and move to C2-C5
        tower_asset = SVGMobject(tower_svg_path).scale(0.3).set_color(C_HIGHLIGHT)
        failed_text = Text("Failed", font_size=28, color=C_HIGHLIGHT)
        math_text = Text(": 1 + 2 = ?", font_size=28, color=C_HIGHLIGHT)
        calc_text = VGroup(failed_text, tower_asset, math_text).arrange(RIGHT, buff=0.1)
        
        self.place_in_area(calc_text, 'C2', 'C5', scale_factor=1.0)
        
        self.play(Write(calc_text))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # One plus two equals three: the error's location!
        self.lecture[1].set_color(C_NEUTRAL)
        self.lecture[2].set_color(C_HIGHLIGHT)
        
        math_result_text = Text(": 1 + 2 = 3", font_size=28, color=C_HIGHLIGHT)
        calc_result = VGroup(failed_text.copy(), tower_asset.copy(), math_result_text).arrange(RIGHT, buff=0.1)
        self.place_in_area(calc_result, 'C2', 'C5', scale_factor=1.0)
        
        self.play(Transform(calc_text, calc_result))
        
        # Highlight bit 3 (Index 2)
        # Using a rectangle to highlight the whole unit
        highlight_rect = bit_boxes[2].copy().set_color(C_HIGHLIGHT).scale(1.2)
        self.play(
            bit_boxes[2].animate.set_color(C_FAIL),
            Create(highlight_rect),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Simply flip bit three to fix the data.
        self.lecture[2].set_color(C_NEUTRAL)
        self.lecture[3].set_color(C_HIGHLIGHT)
        
        # Flip the visual value (0 -> 1)
        new_bit_label = Text("1", font_size=20, color=C_PASS).move_to(bit_boxes[2].get_center())
        
        self.play(
            Transform(bit_labels[2], new_bit_label),
            bit_boxes[2].animate.set_color(C_PASS),
            FadeOut(highlight_rect),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Error correction achieved without a re-send.
        self.lecture[3].set_color(C_NEUTRAL)
        self.lecture[4].set_color(C_HIGHLIGHT)
        
        # Issue 25 & 36: Use checkmark asset and move to E2-E5
        success_msg = Text("Error Corrected!", font_size=32, color=C_PASS)
        checkmark = SVGMobject(checkmark_svg_path).scale(0.4).set_color(C_PASS)
        success_group = VGroup(success_msg, checkmark).arrange(RIGHT, buff=0.2)
        
        self.place_in_area(success_group, 'E2', 'E5', scale_factor=1.1)
        
        self.play(
            p1_indicator[0].animate.set_color(C_PASS),
            p2_indicator[0].animate.set_color(C_PASS),
            FadeOut(calc_text),
            run_time=1
        )
        self.play(FadeIn(success_group))
        self.wait(2)

        # Final state cleanup
        self.lecture[4].set_color(C_NEUTRAL)
