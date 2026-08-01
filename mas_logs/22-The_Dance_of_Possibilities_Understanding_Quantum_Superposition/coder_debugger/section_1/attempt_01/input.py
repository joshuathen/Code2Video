from manim import *
import numpy as np
import os
from pathlib import Path

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
        # Setup layout
        title_text = "The Classical Binary World (Prerequisite)"
        lecture_lines = [
            'Our classical world is built on definite states.',
            'A switch is either "On" or "Off".',
            'There is no middle ground between these two.',
            'If you leave, the state remains exactly the same.',
            'Things exist in one single place at once.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Assets for animation ===
        
        # Line 1: DEFINITE text and box
        definite_text = Text("DEFINITE", font_size=32, weight=BOLD, color=WHITE)
        definite_box = SurroundingRectangle(definite_text, color=WHITE, buff=0.2)
        definite_group = VGroup(definite_text, definite_box)
        self.place_at_grid(definite_group, 'A4')
        
        # Line 2-3: Switch elements
        # Issue 26 Fix: switch_case area B3-E3 (vertical line representation)
        switch_case = Rectangle(height=3.2, width=0.4, color="#888888", fill_opacity=0.2)
        self.place_in_area(switch_case, 'B3', 'E3')
        
        # Toggle is a square per prompt description
        toggle = Square(side_length=0.4, color=WHITE, fill_opacity=1)
        self.place_at_grid(toggle, 'E3')
        
        # Issue 27 Fix: scale_factor=1.2 for labels
        off_label = Text("Off", font_size=24, color=WHITE)
        self.place_at_grid(off_label, 'E4', scale_factor=1.2)
        
        on_label = Text("On", font_size=24, color=WHITE)
        self.place_at_grid(on_label, 'B4', scale_factor=1.2)
        
        state_num = Text("0", font_size=48, color=WHITE)
        self.place_at_grid(state_num, 'A3')

        # Line 4: Person icon construction
        p_head = Circle(radius=0.15, color=WHITE)
        p_body = Line(DOWN*0.15, DOWN*0.5, color=WHITE)
        p_arms = Line(LEFT*0.2, RIGHT*0.2, color=WHITE).move_to(p_body.get_center())
        p_legs = VGroup(Line(ORIGIN, DOWN*0.3+LEFT*0.15), Line(ORIGIN, DOWN*0.3+RIGHT*0.15)).next_to(p_body, DOWN, buff=0)
        person = VGroup(p_head, p_body, p_arms, p_legs)
        # Issue 28 Fix: Position at C6 with scale 0.8
        self.place_at_grid(person, 'C6', scale_factor=0.8)

        # Line 5: Binary positions
        box1 = Square(side_length=0.8, color=WHITE)
        box2 = Square(side_length=0.8, color=WHITE)
        self.place_at_grid(box1, 'F2')
        self.place_at_grid(box2, 'F4')
        dot = Circle(radius=0.25, color=WHITE, fill_opacity=1)
        self.place_at_grid(dot, 'F2')

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(definite_group), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(
            Create(switch_case),
            FadeIn(toggle),
            FadeIn(off_label),
            FadeIn(on_label),
            Write(state_num),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # New state label "1" in #FFFF00
        state_num_new = Text("1", font_size=48, color="#FFFF00")
        self.place_at_grid(state_num_new, 'A3')
        
        # Fast jump to "On" to simulate binary change
        self.play(
            toggle.animate.move_to(self.grid['B3']),
            Transform(state_num, state_num_new),
            run_time=0.8
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.play(FadeIn(person))
        # Move person to the right edge and fade out
        self.play(
            person.animate.shift(RIGHT*1.5).set_opacity(0),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Represent objects existing in one definite place
        self.play(
            Create(box1),
            Create(box2),
            run_time=1
        )
        self.play(FadeIn(dot), run_time=0.5)
        self.wait(2)

        # Cleanup
        self.lecture[4].set_color(WHITE)
