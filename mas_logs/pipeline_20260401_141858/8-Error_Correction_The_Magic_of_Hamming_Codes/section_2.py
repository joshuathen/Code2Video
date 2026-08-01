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
        lecture_lines_text = [
            "Parity bits help detect if a bit has flipped.",
            "Even parity makes the total number of ones even.",
            "Parity detects errors but cannot locate the specific bit."
        ]
        self.setup_layout("Prerequisite: The Simple Parity Bit", lecture_lines_text)
        
        # Colors for mapping
        HIGHLIGHT_COLOR = YELLOW
        PARITY_COLOR = "#00FF00"  # Green
        ERROR_COLOR = "#FF0000"   # Red

        # Assets
        device_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/device.svg"
        tool_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/tool.svg"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # Device Icon
        device_icon = SVGMobject(device_asset_path, color=WHITE)
        self.place_at_grid(device_icon, "A1", scale_factor=0.6)
        self.play(FadeIn(device_icon))

        # Create initial data bits '101'
        b1 = Text("1", font_size=40)
        b2 = Text("0", font_size=40)
        b3 = Text("1", font_size=40)
        self.place_at_grid(b1, "B2")
        self.place_at_grid(b2, "B3")
        self.place_at_grid(b3, "B4")
        
        data_bits = VGroup(b1, b2, b3)
        self.play(Write(data_bits))
        
        # Append parity bit '0' in green box
        p_bit = Text("0", font_size=40, color=PARITY_COLOR)
        self.place_at_grid(p_bit, "B5")
        p_box = SurroundingRectangle(p_bit, color=PARITY_COLOR, buff=0.1)
        
        # Fix Issue 32: scale_factor=0.7 for parity label
        p_label = Text("Parity", font_size=16, color=PARITY_COLOR)
        self.place_at_grid(p_label, "A5", scale_factor=0.7)
        
        self.play(Create(p_box), Write(p_bit), Write(p_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(PARITY_COLOR)
        )
        
        # Show logic: 1 + 0 + 1 = 2 (Even)
        logic_text = Text("1 + 0 + 1 = 2 (Even)", font_size=24, color=PARITY_COLOR)
        self.place_in_area(logic_text, "D2", "D5")
        
        self.play(FadeIn(logic_text))
        self.play(Indicate(p_bit, color=PARITY_COLOR))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(ERROR_COLOR)
        )
        
        # Simulate transmission: flip bit 2 (0 -> 1)
        flipped_b2 = Text("1", font_size=40, color=ERROR_COLOR)
        self.place_at_grid(flipped_b2, "B3")
        
        # Checker Icon
        checker_icon = SVGMobject(tool_asset_path, color=ERROR_COLOR)
        self.place_at_grid(checker_icon, "D1", scale_factor=0.6)
        
        # Change whole string to red
        received_bits = VGroup(b1, flipped_b2, b3, p_bit)
        
        self.play(
            Transform(b2, flipped_b2),
            p_box.animate.set_color(ERROR_COLOR),
            p_label.animate.set_color(ERROR_COLOR),
            FadeOut(logic_text),
            FadeOut(device_icon),
            FadeIn(checker_icon)
        )
        self.play(received_bits.animate.set_color(ERROR_COLOR))

        # Show checker counting bits
        error_logic = Text("1 + 1 + 1 + 0 = 3 (Odd!)", font_size=24, color=ERROR_COLOR)
        self.place_in_area(error_logic, "D2", "D5")
        
        # Fix Issue 31: Place error_indicator at E2-E5 with scale_factor=0.7
        error_indicator = Text("ERROR DETECTED", font_size=28, color=ERROR_COLOR, weight=BOLD)
        self.place_in_area(error_indicator, "E2", "E5", scale_factor=0.7)
        
        self.play(FadeIn(error_logic))
        self.play(Write(error_indicator))
        
        # Highlight uncertainty: "?" over each bit
        q_marks = VGroup(*[Text("?", font_size=30, color=WHITE) for _ in range(4)])
        self.place_at_grid(q_marks[0], "C2")
        self.place_at_grid(q_marks[1], "C3")
        self.place_at_grid(q_marks[2], "C4")
        self.place_at_grid(q_marks[3], "C5")
        
        # Fix Issue 30: Place uncertainty_text at F2-F5 with scale_factor=0.6
        uncertainty_text = Text("Which bit is wrong?", font_size=20, color=WHITE)
        self.place_in_area(uncertainty_text, "F2", "F5", scale_factor=0.6)
        
        self.play(LaggedStart(*[FadeIn(q) for q in q_marks], lag_ratio=0.2))
        self.play(FadeIn(uncertainty_text))
        
        self.wait(2)
