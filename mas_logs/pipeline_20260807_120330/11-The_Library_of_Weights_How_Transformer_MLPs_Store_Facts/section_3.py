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

class Section3Scene(TeachingScene):
    def construct(self):
        # Fetching data from storyboard
        title_text = "Prerequisite: The Key-Value Analogy"
        lecture_lines = [
            "Think of the MLP as a vast dictionary.",
            "It uses the concept of Key-Value pairs.",
            "An input vector acts as a query key.",
            "If it aligns with a stored key, it activates.",
            "This alignment \"unlocks\" a specific, meaningful value vector."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors from Animation Description
        COLOR_DICT = "#FFFFFF"
        COLOR_LABEL = "#FFA500"
        COLOR_QUERY = "#00FFFF"
        COLOR_PULSE = "#FFFF00"
        COLOR_VALUE = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # "Think of the MLP as a vast dictionary."
        self.lecture[0].set_color(COLOR_DICT)
        # Asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/dictionary.svg]
        dict_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/dictionary.svg")
        dict_svg.set_color(COLOR_DICT)
        # Fix: Update Line 68 to self.place_in_area(dict_label, 'A1', 'A6')
        self.place_in_area(dict_svg, 'A1', 'A6', scale_factor=0.6)
        
        self.play(DrawBorderThenFill(dict_svg))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "It uses the concept of Key-Value pairs."
        self.lecture[1].set_color(COLOR_LABEL)
        
        # Key and Value vectors
        key_vector = Arrow(DOWN, UP, color=COLOR_LABEL, buff=0)
        value_vector = Arrow(DOWN, UP, color=COLOR_LABEL, buff=0)
        
        # Labels
        key_label = Text("Key", font_size=20, color=COLOR_LABEL)
        value_label = Text("Value", font_size=20, color=COLOR_LABEL)
        
        # Fix Layout based on Issue 49:
        # key_arrow at 'D2', scale 0.8
        self.place_at_grid(key_vector, 'D2', scale_factor=0.8)
        # key_label at 'E2'
        self.place_at_grid(key_label, 'E2')
        # value_arrow at 'D5', scale 0.8
        self.place_at_grid(value_vector, 'D5', scale_factor=0.8)
        # value_label at 'E5'
        self.place_at_grid(value_label, 'E5')
        
        self.play(
            FadeIn(key_vector), Write(key_label),
            FadeIn(value_vector), Write(value_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "An input vector acts as a query key."
        self.lecture[2].set_color(COLOR_QUERY)
        
        query_vector = Arrow(DOWN, UP, color=COLOR_QUERY, buff=0)
        query_label = Text("Query", font_size=20, color=COLOR_QUERY)
        
        # Fix Layout based on Issue 49:
        # input_arrow at 'C1', scale 0.8
        self.place_at_grid(query_vector, 'C1', scale_factor=0.8)
        # input_label at 'B1'
        self.place_at_grid(query_label, 'B1')
        
        # Initially misaligned (rotated)
        query_vector.rotate(-PI/4)
        
        self.play(FadeIn(query_vector), Write(query_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "If it aligns with a stored key, it activates."
        self.lecture[3].set_color(COLOR_PULSE)
        
        # Move query to key and align
        self.play(
            query_vector.animate.move_to(key_vector.get_center()).rotate(PI/4),
            query_label.animate.next_to(key_vector, LEFT, buff=0.2),
            run_time=1.5
        )
        
        # Pulse effect
        pulse = Circle(radius=0.5, color=COLOR_PULSE).move_to(key_vector.get_center())
        self.play(
            pulse.animate.scale(3).set_stroke(opacity=0),
            run_time=0.8,
            rate_func=rate_functions.ease_out_quad
        )
        self.remove(pulse)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "This alignment \"unlocks\" a specific, meaningful value vector."
        self.lecture[4].set_color(COLOR_VALUE)
        
        # Highlight and scale up Value vector
        self.play(
            value_vector.animate.scale(1.5).set_color(COLOR_VALUE),
            value_label.animate.scale(1.2).set_color(COLOR_VALUE),
        )
        
        flash = Flash(value_vector, color=COLOR_VALUE, flash_radius=1.0)
        self.play(flash)
        self.wait(2)
