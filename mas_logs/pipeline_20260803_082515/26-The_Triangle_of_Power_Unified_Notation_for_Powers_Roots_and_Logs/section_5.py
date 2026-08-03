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
        # Setup layout with title and lecture lines
        self.setup_layout("Operation 2: Solving for the Base (Roots)", [
            "Solving for the Base replaces the radical symbol.",
            "Root the Rabbit digs for the unknown bottom-left.",
            "If Exponent is 3 and Result 64, Base is 4."
        ])
        
        # Define shared positions using grid
        # Updated triangle vertices based on Issue #33 and #34 to shift everything right
        # Top vertex centered between B4 and B5
        top_v = (self.grid["B4"] + self.grid["B5"]) / 2
        # Bottom left at E3
        bl_v = self.grid["E3"]
        # Bottom right at E6
        br_v = self.grid["E6"]

        # === Animation for Lecture Line 1 ===
        # Highlight Line 1 with the color of the '?'
        self.lecture[0].set_color("#FF4500")
        
        # Triangle and initial labels
        triangle = Polygon(top_v, br_v, bl_v, color=WHITE)
        
        # Exponent (Fix for Issue #34: Move to A4-A5)
        exponent = Text("3", color=WHITE)
        self.place_in_area(exponent, "A4", "A5", scale_factor=0.8)
        
        # Result (Fix for Issue #33: Move to F6)
        result = Text("64", color=WHITE)
        self.place_at_grid(result, "F6", scale_factor=0.8)
        
        # Base Unknown (Fix for Issue #33: Move to F3)
        base_unknown = Text("?", color="#FF4500")
        self.place_at_grid(base_unknown, "F3", scale_factor=0.8)
        
        self.play(Create(triangle))
        self.play(Write(exponent), Write(result), Write(base_unknown))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Reset previous and highlight Line 2 (Rabbit is White)
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(WHITE)
        
        # Rabbit icon (Root the Rabbit)
        # Fix for Issue #32: Start at E2 instead of E1
        rabbit = VGroup(
            Ellipse(width=0.3, height=0.4, color=WHITE, fill_opacity=1),
            Ellipse(width=0.1, height=0.3, color=WHITE, fill_opacity=1).shift(UP*0.15 + LEFT*0.07),
            Ellipse(width=0.1, height=0.3, color=WHITE, fill_opacity=1).shift(UP*0.15 + RIGHT*0.07)
        )
        self.place_at_grid(rabbit, "E2", scale_factor=0.8)
        
        self.play(FadeIn(rabbit))
        # Digging animation near the bottom-left vertex (E3)
        dig_pos = bl_v + DOWN * 0.2
        self.play(rabbit.animate.move_to(dig_pos), run_time=0.4)
        for _ in range(3):
            self.play(rabbit.animate.shift(UP * 0.05), run_time=0.1)
            self.play(rabbit.animate.shift(DOWN * 0.05), run_time=0.1)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Reset previous and highlight Line 3 (Gold)
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FFD700")
        
        # Base value 4
        base_value = Text("4", color="#FFD700")
        # Position base_value at the same spot as base_unknown (F3)
        base_value.move_to(base_unknown.get_center())
        
        self.play(
            Transform(base_unknown, base_value),
            rabbit.animate.shift(LEFT * 1.5)
        )
        self.wait(2)
