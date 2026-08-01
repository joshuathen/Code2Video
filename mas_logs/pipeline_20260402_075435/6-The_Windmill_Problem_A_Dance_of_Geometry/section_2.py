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
        # Initialize Scene
        lecture_lines = [
            'Avoid setups where three stars align.',
            'We assume stars are in general position.',
            'This ensures the laser hits only one at once.'
        ]
        self.setup_layout("Prerequisite: Rotation and Pivot Points", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Avoid setups where three stars align (Invalid case)
        color_inv = "#FF0000"
        self.play(self.lecture[0].animate.set_color(color_inv))
        
        bad_stars = VGroup(
            Dot(self.grid["D2"], color=color_inv),
            Dot(self.grid["D4"], color=color_inv),
            Dot(self.grid["D6"], color=color_inv)
        )
        bad_line = Line(self.grid["D1"], self.grid["D6"], color=color_inv)
        
        label_inv = Text("Invalid", font_size=20, color=color_inv)
        # Fix 31: Move label to E4 to avoid overlap and bottom edge
        self.place_at_grid(label_inv, "E4", scale_factor=0.8)
        
        self.play(Create(bad_stars), Create(bad_line), Write(label_inv))
        self.wait(2)
        self.play(FadeOut(bad_stars, bad_line, label_inv))

        # === Animation for Lecture Line 2 ===
        # We assume stars are in general position (Scattered)
        color_gen = "#FFFFFF"
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(color_gen)
        )
        
        stars_gen = VGroup(
            Dot(self.grid["C2"]),
            Dot(self.grid["C5"]),
            Dot(self.grid["D3"]),
            Dot(self.grid["D4"]),
            Dot(self.grid["D6"])
        ).set_color(color_gen)
        
        # Static line hitting only one star
        line_gen = Line(LEFT*1.5, RIGHT*1.5, color=color_gen).rotate(15*DEGREES)
        self.place_at_grid(line_gen, "D3")
        
        label_gen = Text("General Position", font_size=20, color=color_gen)
        # Fix 30: Move label to area A3-B5 to avoid overlap with stars
        self.place_in_area(label_gen, 'A3', 'B5', scale_factor=0.7)
        
        self.play(Create(stars_gen), Create(line_gen), Write(label_gen))
        self.wait(2)
        self.play(FadeOut(stars_gen, line_gen, label_gen))

        # === Animation for Lecture Line 3 ===
        # Animation of the line successfully pivoting
        color_pivot = BLUE_C
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(color_pivot)
        )
        
        p1_pos = self.grid["B2"]
        p2_pos = self.grid["B5"]
        p1 = Dot(p1_pos, color=color_pivot)
        p2 = Dot(p2_pos, color=color_pivot)
        
        angle_tracker = ValueTracker(-40 * DEGREES)
        current_pivot = [p1_pos]
        
        rot_line = Line(LEFT*1.8, RIGHT*1.8, color=color_pivot)
        # Position is handled dynamically by updater
        rot_line.add_updater(lambda m: m.set_angle(angle_tracker.get_value()).move_to(current_pivot[0]))
        
        self.add(p1, p2, rot_line)
        self.play(angle_tracker.animate.set_value(0), run_time=1.5, rate_func=linear)
        
        # Transition pivot point to p2
        current_pivot[0] = p2_pos
        self.play(angle_tracker.animate.set_value(40 * DEGREES), run_time=1.5, rate_func=linear)
        self.wait(2)
