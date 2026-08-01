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
        # Colors
        KEY_COLOR = "#FF4500"
        VALUE_COLOR = "#1E90FF"
        
        # Setup
        lecture_lines = [
            "Knowledge relies on a system of keys and values.",
            "A key represents a specific pattern to recognize.",
            "The value is the information associated with that pattern."
        ]
        self.setup_layout("Prerequisite: Key-Value Memory Systems", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Dictionary Icon [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/dictionary.svg]
        dict_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/dictionary.svg")
        dict_asset.set_color(WHITE)
        # Issue 44: Fix positioning to C2-E4
        self.place_in_area(dict_asset, "C2", "E4", scale_factor=1.2)
        
        key_label = Text("Key", color=KEY_COLOR, font_size=24)
        value_label = Text("Value", color=VALUE_COLOR, font_size=24)
        
        self.place_at_grid(key_label, "B2", scale_factor=1.0)
        # Issue 45: Fix positioning to E5
        self.place_at_grid(value_label, "E5", scale_factor=1.0)
        
        self.play(DrawBorderThenFill(dict_asset))
        self.play(Write(key_label), Write(value_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW)
        )
        
        # Vending Machine [Asset: /mmfs1/data/home/jthen/Code2Video/assets/icon/vending.svg]
        vending_asset = SVGMobject("/mmfs1/data/home/jthen/Code2Video/assets/icon/vending.svg")
        vending_asset.set_color(GREY_A)
        self.place_in_area(vending_asset, "A2", "F5", scale_factor=1.8)
        
        self.play(
            FadeOut(dict_asset), FadeOut(key_label), FadeOut(value_label),
            FadeIn(vending_asset)
        )
        
        # Input 'A1' (Key)
        input_key = Text("A1", color=KEY_COLOR, font_size=32)
        # Issue 46: Fix initial positioning to B2
        self.place_at_grid(input_key, "B2") 
        
        # Simulate slot and output center relative to machine
        slot_center = vending_asset.get_center() + UP*0.5
        bin_center = vending_asset.get_center() + DOWN*1.2
        
        self.play(input_key.animate.move_to(slot_center))
        self.wait(0.5)
        
        # Output 'Chips' (Value)
        output_val = Text("Chips", color=VALUE_COLOR, font_size=32)
        output_val.move_to(slot_center)
        
        self.play(
            output_val.animate.move_to(bin_center),
            input_key.animate.set_opacity(0)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW)
        )
        
        # Matrix Representation
        matrix_rows = 4
        matrix_cols = 5
        cells = VGroup()
        for r in range(matrix_rows):
            for c in range(matrix_cols):
                rect = Square(side_length=0.6, stroke_width=1, color=WHITE)
                rect.move_to(np.array([c*0.6, -r*0.6, 0]))
                cells.add(rect)
        
        matrix_group = cells.center()
        self.place_in_area(matrix_group, "B2", "E5", scale_factor=1.2)
        
        # Highlight a row as a [Key | Value] pattern
        highlight_row = VGroup(*[cells[i] for i in range(matrix_cols)])
        highlight_label = Text("Pattern Row", font_size=20, color=YELLOW).next_to(highlight_row, UP)
        
        # Final transformation
        self.play(
            FadeOut(vending_asset),
            FadeOut(output_val),
            FadeIn(matrix_group)
        )
        
        # Show key/value as components of a row
        row_key_part = Text("K", color=KEY_COLOR, font_size=24).move_to(cells[1].get_center())
        row_val_part = Text("V", color=VALUE_COLOR, font_size=24).move_to(cells[3].get_center())
        
        self.play(
            highlight_row.animate.set_stroke(YELLOW, width=4),
            Write(highlight_label),
            Write(row_key_part),
            Write(row_val_part)
        )
        
        self.wait(2)
        self.play(self.lecture[2].animate.set_color(WHITE))
