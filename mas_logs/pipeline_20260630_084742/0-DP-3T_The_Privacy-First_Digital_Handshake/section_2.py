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
        # Setup the scene with title and lecture lines
        lecture_lines = [
            'Cryptographic hashing turns inputs into unique strings.',
            'These one-way functions cannot be reversed.',
            'A secret seed generates unpredictable rolling codes.'
        ]
        self.setup_layout("Prerequisite: One-Way Hash Functions", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first line
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1:3].animate.set_color(GRAY))
        
        # Create Blender (Hash Function)
        blender_body = Polygon(
            [-0.8, 1, 0], [0.8, 1, 0], [0.5, -1, 0], [-0.5, -1, 0],
            color=WHITE, fill_opacity=0.2
        )
        blender_label = Text("Hash Function", font_size=20, color=WHITE)
        blender = VGroup(blender_body, blender_label)
        # Fix Issue 36: Move blender to C5-D6
        self.place_in_area(blender, "C5", "D6", scale_factor=0.8)
        blender_label.next_to(blender_body, DOWN, buff=0.2)
        
        self.play(Create(blender_body), Write(blender_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight second line
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[0].animate.set_color(GRAY), self.lecture[2].animate.set_color(GRAY))
        
        # Create a "One-Way" indicator (Red Arrow)
        # Adjusted arrow to align with the new column (shifted from 3 to 4)
        one_way_arrow = Arrow(
            start=self.grid["B4"], end=self.grid["E4"], 
            color=RED, stroke_width=8, buff=0.1
        ).shift(LEFT * 0.2)
        one_way_label = Text("ONE-WAY", font_size=18, color=RED).rotate(90*DEGREES).next_to(one_way_arrow, LEFT, buff=0.1)
        
        self.play(GrowArrow(one_way_arrow), Write(one_way_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight third line
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[0:2].animate.set_color(GRAY))
        
        # Seed Block
        seed_box = Square(side_length=0.6, color=YELLOW, fill_opacity=0.8)
        seed_text = Text("Seed", font_size=16, color=BLACK)
        seed = VGroup(seed_box, seed_text)
        # Fix Issue 37: Move seed to B5, scale 0.7
        self.place_at_grid(seed, "B5", scale_factor=0.7)
        
        # Rolling Codes (Output)
        code_strings = ["0x8F2A", "0x3C1B", "0x9E7D"]
        codes = VGroup(*[
            VGroup(
                RoundedRectangle(height=0.4, width=1.2, corner_radius=0.1, color=GREEN, fill_opacity=0.8),
                Text(s, font_size=16, color=BLACK)
            ) for s in code_strings
        ]).arrange(DOWN, buff=0.2)
        # Fix Issue 35: Move codes to F5, scale 0.7
        self.place_at_grid(codes, "F5", scale_factor=0.7)
        
        # Animation: Seed enters Blender, Codes come out
        self.play(FadeIn(seed, shift=DOWN))
        self.play(seed.animate.move_to(blender_body.get_center()).set_opacity(0), run_time=1)
        
        # Shake blender
        self.play(blender_body.animate.shift(LEFT*0.1), run_time=0.05)
        self.play(blender_body.animate.shift(RIGHT*0.2), run_time=0.05)
        self.play(blender_body.animate.shift(LEFT*0.1), run_time=0.05)
        
        # Flow out codes
        for code in codes:
            self.play(
                FadeIn(code, target_position=blender_body.get_center()),
                code.animate.move_to(code.get_center()),
                run_time=0.5
            )
        
        self.wait(2)
