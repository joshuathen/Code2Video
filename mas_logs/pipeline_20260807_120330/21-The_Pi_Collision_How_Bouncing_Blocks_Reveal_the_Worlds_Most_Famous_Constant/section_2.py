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
        lecture_lines = [
            "We count every collision between blocks and the wall.",
            "With equal masses, we observe three total collisions.",
            "Increase the large mass by a factor of 100.",
            "Now we count 31 collisions in total.",
            "The digits of Pi begin to appear magically."
        ]
        self.setup_layout("The Pattern Emerges", lecture_lines)
        
        # Colors for lines
        colors = ["#88C0D0", "#A3BE8C", "#EBCB8B", "#D08770", "#B48EAD"]
        highlight_color = "#FFFF00"

        # === Animation for Lecture Line 1 ===
        # "We count every collision between blocks and the wall."
        self.lecture[0].set_color(colors[0])
        
        header_mass = Text("Mass Ratio", font_size=24, color=colors[0])
        header_coll = Text("Collisions", font_size=24, color=colors[0])
        
        # Resolving Issue 23: Fix overlapping headers
        self.place_in_area(header_mass, "B1", "B2", scale_factor=0.6)
        self.place_in_area(header_coll, "B5", "B6", scale_factor=0.6)
        
        # Table lines based on grid positions
        y_h = (self.grid["B1"][1] + self.grid["C1"][1]) / 2
        h_line = Line(
            [self.grid["B1"][0] - 0.5, y_h, 0],
            [self.grid["B6"][0] + 0.5, y_h, 0],
            color=colors[0]
        )
        
        x_v = (self.grid["B3"][0] + self.grid["B4"][0]) / 2
        v_line = Line(
            [x_v, self.grid["B1"][1] + 0.3, 0],
            [x_v, self.grid["E1"][1] - 0.3, 0],
            color=colors[0]
        )

        self.play(Write(header_mass), Write(header_coll), Create(h_line), Create(v_line))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "With equal masses, we observe three total collisions."
        self.lecture[1].set_color(colors[1])
        
        ratio_1 = MathTex("1 : 1", color=colors[1])
        count_1 = MathTex("3", color=colors[1])
        
        self.place_in_area(ratio_1, "C1", "C3", scale_factor=0.8)
        self.place_in_area(count_1, "C4", "C6", scale_factor=0.8)
        
        self.play(FadeIn(ratio_1), Write(count_1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Increase the large mass by a factor of 100."
        self.lecture[2].set_color(colors[2])
        
        ratio_2 = MathTex("100 : 1", color=colors[2])
        self.place_in_area(ratio_2, "D1", "D3", scale_factor=0.8)
        
        self.play(FadeIn(ratio_2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Now we count 31 collisions in total."
        self.lecture[3].set_color(colors[3])
        
        count_2 = MathTex("31", color=colors[3])
        self.place_in_area(count_2, "D4", "D6", scale_factor=0.8)
        
        self.play(Write(count_2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "The digits of Pi begin to appear magically."
        self.lecture[4].set_color(colors[4])
        
        ratio_3 = MathTex("10,000 : 1", color=colors[4])
        count_3 = MathTex("314", color=colors[4])
        
        # Resolving Issue 24: Fix cramped ratio_3
        self.place_in_area(ratio_3, "E1", "E3", scale_factor=0.65)
        # Resolving Issue 25: Fix misaligned count_3
        self.place_at_grid(count_3, "E5", scale_factor=0.8)
        
        self.play(FadeIn(ratio_3), Write(count_3))
        self.wait(1)
        
        # Highlight sequence (digits 3, 1, 4)
        self.play(
            count_1.animate.set_color(highlight_color),
            count_2.animate.set_color(highlight_color),
            count_3.animate.set_color(highlight_color),
            run_time=1.5
        )
        
        pi_text = MathTex(r"\pi \approx 3.14159...", color=highlight_color)
        # Summary text placed in Row F, keeping scale moderate
        self.place_in_area(pi_text, "F1", "F6", scale_factor=0.8)
        self.play(FadeIn(pi_text))
        self.wait(2)
