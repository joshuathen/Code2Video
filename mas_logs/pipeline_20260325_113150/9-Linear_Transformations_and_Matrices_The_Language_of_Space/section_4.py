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

class Section4Scene(TeachingScene):
    def construct(self):
        # Initial setup of the lecture layout
        self.setup_layout(
            "The Matrix: A Storage Device",
            [
                "A matrix packages the landing spots of basis vectors.",
                "The first column shows where i-hat landed.",
                "The second column shows where j-hat landed."
            ]
        )

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        # Manually construct the matrix using Text and VGroup
        matrix_data = [["3", "2"], ["-2", "1"]]
        entry_mobjects = [Text(item, font_size=42) for row in matrix_data for item in row]
        entries_group = VGroup(*entry_mobjects).arrange_in_grid(rows=2, cols=2, buff=0.8)
        
        l_bracket = Text("[", font_size=60)
        r_bracket = Text("]", font_size=60)
        
        # Adjust height of brackets to match the entries
        l_bracket.stretch_to_fit_height(entries_group.height + 0.6)
        r_bracket.stretch_to_fit_height(entries_group.height + 0.6)
        l_bracket.next_to(entries_group, LEFT, buff=0.2)
        r_bracket.next_to(entries_group, RIGHT, buff=0.2)
        
        brackets = VGroup(l_bracket, r_bracket).set_color(WHITE)
        
        # Construct the matrix-like object as a VGroup
        matrix_obj = VGroup(entries_group, brackets)
        
        # Helper accessors
        matrix_obj.get_entries = lambda: entries_group
        matrix_obj.get_brackets = lambda: brackets
        
        # Hide the numbers initially to animate them per column
        for entry in matrix_obj.get_entries():
            entry.set_opacity(0)
            
        # Place the matrix in the designated right-side area
        # Fixed scaling and positioning as per issues 34, 35, 36
        self.place_in_area(matrix_obj, "B3", "E4", scale_factor=0.85)
        
        self.play(Create(matrix_obj.get_brackets()))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Using coordinated Red color #FC6255
        self.play(self.lecture[1].animate.set_color("#FC6255"))
        
        # Column 1 entries: index 0 and 2 in the flattened 2x2 list [a11, a12, a21, a22]
        entry_1_1 = matrix_obj.get_entries()[0]
        entry_2_1 = matrix_obj.get_entries()[2]
        
        entry_1_1.set_color("#FC6255").set_opacity(1)
        entry_2_1.set_color("#FC6255").set_opacity(1)
        
        self.play(
            Write(entry_1_1),
            Write(entry_2_1)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Using coordinated Green color #83C167
        self.play(self.lecture[2].animate.set_color("#83C167"))
        
        # Column 2 entries: index 1 and 3
        entry_1_2 = matrix_obj.get_entries()[1]
        entry_2_2 = matrix_obj.get_entries()[3]
        
        entry_1_2.set_color("#83C167").set_opacity(1)
        entry_2_2.set_color("#83C167").set_opacity(1)
        
        self.play(
            Write(entry_1_2),
            Write(entry_2_2)
        )
        self.wait(2)
