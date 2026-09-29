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
        self.setup_layout("Prerequisite: Parity and Binary States", [
            "Parity tracks if coin sums are even or odd.",
            "Heads is one, tails is zero in binary.",
            "Toggling a coin flips the board's total parity."
        ])
        
        # Assets
        coin_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coin.svg")
        
        # Build binary coin group (H, T, H)
        def create_coin(state_text):
            coin = coin_icon.copy().scale(0.3)
            label = Text(state_text, font_size=20)
            return VGroup(coin, label).arrange(DOWN, buff=0.1)

        coin_group = VGroup(
            create_coin("1"),
            create_coin("0"),
            create_coin("1")
        ).arrange(RIGHT, buff=0.4)
        
        self.place_at_grid(coin_group, "B4", scale_factor=0.8)
        
        parity_text = Text("Parity: 0 (Even)", font_size=24, color="#E0FFFF")
        self.place_at_grid(parity_text, "C4", scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#E0FFFF"))
        self.play(FadeIn(coin_group), Write(parity_text))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#ADD8E6"))
        # Add light blue pulses to coins
        self.play(Indicate(coin_group, color=BLUE_B), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#90EE90"))
        
        # Toggle coin (H to T)
        new_c = create_coin("0")
        new_c.move_to(coin_group[0].get_center())
        
        self.play(
            ReplacementTransform(coin_group[0], new_c),
            parity_text.animate.set_text("Parity: 1 (Odd)"),
            run_time=1.5
        )
        self.wait(2)
