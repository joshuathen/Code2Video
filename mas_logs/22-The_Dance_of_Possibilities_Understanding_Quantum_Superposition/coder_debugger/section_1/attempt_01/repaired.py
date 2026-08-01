from manim import *
import numpy as np
import os
from pathlib import Path

class TeachingScene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        
        # Ensure the media/texts directory exists and is a directory to prevent FileExistsError
        text_dir = Path(config.get_dir("text_dir"))
        if text_dir.exists() and not text_dir.is_dir():
            os.remove(text_dir)
        os.makedirs(text_dir, exist_ok=True)

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

        # Assets for animation
        switch_case = Rectangle(height=4.5, width=2.0, color="#888888", fill_opacity=0.2)
        self.place_in_area(switch_case, 'B2', 'E4')
        
        toggle = Circle(radius=0.3, color=WHITE, fill_opacity=1)
        self.place_at_grid(toggle, 'E3')
        
        off_label = Text("Off", font_size=24, color=WHITE)
        self.place_at_grid(off_label, 'E4')
        
        on_label = Text("On", font_size=24, color=WHITE)
        self.place_at_grid(on_label, 'B4')
        
        state_num = Text("0", font_size=48, color=WHITE)
        self.place_at_grid(state_num, 'A3')

        # Person icon construction
        p_head = Circle(radius=0.15, color=WHITE)
        p_body = Line(DOWN*0.15, DOWN*0.5, color=WHITE)
        p_arms = Line(LEFT*0.2, RIGHT*0.2, color=WHITE).move_to(p_body.get_center())
        p_legs = VGroup(Line(ORIGIN, DOWN*0.3+LEFT*0.15), Line(ORIGIN, DOWN*0.3+RIGHT*0.15)).next_to(p_body, DOWN, buff=0)
        person = VGroup(p_head, p_body, p_arms, p_legs)
        self.place_at_grid(person, 'C5')

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(switch_case), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        self.play(
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
        
        self.play(
            toggle.animate.move_to(self.grid['B3']),
            Transform(state_num, state_num_new),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)
        self.add(person)
        self.play(
            person.animate.move_to(self.grid['C6']).set_opacity(0),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Highlight the single location of the switch/toggle
        highlight_circle = Circle(radius=0.6, color=YELLOW).move_to(toggle.get_center())
        self.play(Create(highlight_circle), run_time=1)
        self.play(FadeOut(highlight_circle), run_time=1)
        self.wait(2)

        # Cleanup highlight
        self.lecture[4].set_color(WHITE)