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

class Section1Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Prerequisite: The Language of Bits", [
            "Binary numbers rely on powers of two.",
            "Each bit represents a simple decision.",
            "Three bits count from zero to seven."
        ])
        
        # Load asset and create bits
        # The asset is used as a base/background for each square as per storyboard instruction
        asset_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg"
        
        bit1 = VGroup(SVGMobject(asset_path).set_color(WHITE), Square(side_length=0.6, fill_opacity=0.5, color=WHITE))
        bit2 = VGroup(SVGMobject(asset_path).set_color(WHITE), Square(side_length=0.6, fill_opacity=0.5, color=WHITE))
        bit3 = VGroup(SVGMobject(asset_path).set_color(WHITE), Square(side_length=0.6, fill_opacity=0.5, color=WHITE))
        
        bits = VGroup(bit1, bit2, bit3).arrange(RIGHT, buff=0.5)
        
        label1 = Text("Bit 1", font_size=18).next_to(bit1, UP)
        label2 = Text("Bit 2", font_size=18).next_to(bit2, UP)
        label3 = Text("Bit 3", font_size=18).next_to(bit3, UP)
        labels = VGroup(label1, label2, label3)

        bit_box_group = VGroup(bits, labels)
        
        # Positioning according to feedback
        self.place_in_area(bit_box_group, 'B2', 'B4', scale_factor=0.9)
        self.add(bit_box_group)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        
        # Sequence of binary counting
        # 001
        self.play(bit1[1].animate.set_fill(color=YELLOW, opacity=1))
        self.wait(1)
        
        # 010
        self.play(bit1[1].animate.set_fill(color=WHITE, opacity=0.5),
                  bit2[1].animate.set_fill(color=YELLOW, opacity=1))
        self.wait(1)
        
        # 011
        self.play(bit1[1].animate.set_fill(color=YELLOW, opacity=1))
        self.wait(1)
        
        # 100
        self.play(bit1[1].animate.set_fill(color=WHITE, opacity=0.5),
                  bit2[1].animate.set_fill(color=WHITE, opacity=0.5),
                  bit3[1].animate.set_fill(color=YELLOW, opacity=1))
        self.wait(1)
