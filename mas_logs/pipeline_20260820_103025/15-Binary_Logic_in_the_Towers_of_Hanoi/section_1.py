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
        lecture_lines = [
            "Binary represents numbers using switches, ON or OFF.",
            "Each switch corresponds to a power of two.",
            "Thirteen is eight plus four plus one.",
            "This equals one-one-zero-one in binary.",
            "Any decimal number uniquely sums these powers."
        ]
        self.setup_layout("Introduction: The Binary Basics", lecture_lines)
        
        # Load asset
        switch_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/switch.svg")
        bit_circle = Circle(radius=0.5, color="#00FF00", fill_opacity=0.5)

        # === Animation for Lecture Line 1 ===
        # Fixed bit_circle position (Issue 22/33)
        self.place_at_grid(bit_circle, 'C5', scale_factor=0.8)
        self.play(FadeIn(bit_circle))
        self.lecture[0].set_color("#00FF00")

        # === Animation for Lecture Line 2 ===
        # Using asset switch (Issue 17)
        switch_icon.scale(0.5).next_to(bit_circle, UP)
        self.play(FadeIn(switch_icon))
        self.play(bit_circle.animate.set_color("#FFFF00"), run_time=1)
        self.play(bit_circle.animate.set_color("#00FF00"), run_time=1)
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        # Visualization of 8, 4, 1, improved positioning (Issue 23/33)
        eight = MathTex("8", color=WHITE)
        four = MathTex("4", color=WHITE)
        one = MathTex("1", color=WHITE)
        power_labels = VGroup(eight, four, one).arrange(RIGHT, buff=0.5)
        self.place_in_area(power_labels, 'A4', 'A6', scale_factor=0.7)
        self.play(Write(power_labels))
        self.lecture[2].set_color("#00FF00")

        # === Animation for Lecture Line 4 ===
        # Improved positioning (Issue 21/33)
        formula_1101 = Text("1101", color="#00FFFF")
        self.place_in_area(formula_1101, 'A4', 'B6', scale_factor=0.9)
        self.play(Write(formula_1101))
        self.lecture[3].set_color("#00FFFF")

        # === Animation for Lecture Line 5 ===
        self.play(Indicate(formula_1101), run_time=2)
        self.lecture[4].set_color("#00FF00")
        self.wait(2)
