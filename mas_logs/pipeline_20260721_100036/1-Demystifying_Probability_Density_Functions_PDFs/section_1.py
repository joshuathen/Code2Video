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
        self.setup_layout("The Bridge: Discrete vs. Continuous", [
            "- Discrete events involve counting specific individual points.",
            "- Continuous events involve measuring values within a range.",
            "- Like measuring a cat's jump distance precisely."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Show a rolling die landing on 4. [Color die #ADD8E6]
        self.lecture[0].set_color("#ADD8E6")
        
        die_square = Square(side_length=1.2, fill_opacity=1, fill_color="#ADD8E6", stroke_color=WHITE)
        # 4 dots for the number 4
        dot_offset = 0.3
        dots = VGroup(
            Dot(color=BLACK).move_to(die_square.get_center() + dot_offset*UP + dot_offset*LEFT),
            Dot(color=BLACK).move_to(die_square.get_center() + dot_offset*UP + dot_offset*RIGHT),
            Dot(color=BLACK).move_to(die_square.get_center() + dot_offset*DOWN + dot_offset*LEFT),
            Dot(color=BLACK).move_to(die_square.get_center() + dot_offset*DOWN + dot_offset*RIGHT)
        )
        die = VGroup(die_square, dots)
        # Fix Issue 22: Move die to B2, scale 0.9
        self.place_at_grid(die, 'B2', scale_factor=0.9)
        
        self.play(
            FadeIn(die),
            Rotate(die, angle=PI*2, about_point=die.get_center()),
            run_time=1.5
        )
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # Show a ruler measuring a jump distance. [Color marker #FFD700]
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#FFD700")
        )
        
        # Ruler length adjustment to fit area E2-E5 (3 grid units)
        ruler_line = Line(LEFT*1.5, RIGHT*1.5, color=WHITE)
        ticks = VGroup(*[
            Line(UP*0.1, DOWN*0.1, color=WHITE).move_to(ruler_line.get_start() + i*0.3*RIGHT) 
            for i in range(11)
        ])
        ruler = VGroup(ruler_line, ticks)
        # Fix Issue 23: Place ruler in area E2-E5
        self.place_in_area(ruler, 'E2', 'E5', scale_factor=1.0)
        
        marker = Triangle(color="#FFD700", fill_opacity=1).scale(0.15).rotate(PI)
        # Fix Issue 24: Place marker at D3, scale 0.6
        self.place_at_grid(marker, 'D3', scale_factor=0.6) 
        
        self.play(
            FadeIn(ruler),
            FadeIn(marker),
            run_time=1
        )
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        # Ruler turns into a smooth line segment. [Color line #FFFFFF]
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#FFFFFF")
        )
        
        smooth_line = Line(ruler_line.get_start(), ruler_line.get_end(), color=WHITE, stroke_width=4)
        smooth_line.move_to(ruler_line.get_center())
        
        # Animate marker moving to simulate continuous measurement
        target_marker_pos = self.grid['D5']
        
        self.play(
            FadeOut(ticks),
            ReplacementTransform(ruler_line, smooth_line),
            marker.animate.move_to(target_marker_pos),
            run_time=2
        )
        self.wait(3)
