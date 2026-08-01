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
        # Colors
        COLOR_TEXT = WHITE
        COLOR_VALUES = "#00FF00"  # Green
        COLOR_FORMULA = "#FFFF00" # Yellow
        
        self.setup_layout("The Alluring Power of $2^{n-1}$", [
            "One point gives us a single region.",
            "Two points and one cut produce two regions.",
            "Three points with three cuts result in four regions.",
            "Four points give eight, and five points give sixteen.",
            "The sequence follows a perfect power of two pattern."
        ])
        
        # === PRE-CREATION ===
        
        # Table headers
        header_n = Text("n", font_size=24, color=COLOR_TEXT)
        header_regions = Text("Regions", font_size=24, color=COLOR_TEXT)
        self.place_at_grid(header_n, "A2")
        self.place_at_grid(header_regions, "A4")
        
        # Grid lines
        # Horizontal line below headers (centered between Row A and Row B)
        h_line = Line(self.grid["A1"] + DOWN*0.5, self.grid["A6"] + DOWN*0.5, color=COLOR_TEXT)
        # Vertical line between columns (centered between Col 2 and Col 4)
        v_line = Line(self.grid["A3"] + UP*0.5, self.grid["F3"] + DOWN*0.5, color=COLOR_TEXT)
        
        # Data rows
        rows_data = [
            ("1", "1", "B"),
            ("2", "2", "C"),
            ("3", "4", "D"),
            ("4", "8", "E"),
            ("5", "16", "F")
        ]
        
        row_mobjects = []
        for n_val, r_val, row_char in rows_data:
            n_mob = Text(n_val, font_size=24, color=COLOR_VALUES)
            r_mob = Text(r_val, font_size=24, color=COLOR_VALUES)
            self.place_at_grid(n_mob, f"{row_char}2")
            self.place_at_grid(r_mob, f"{row_char}4")
            row_mobjects.append((n_mob, r_mob))
            
        # Formula pattern - Using MathTex for proper exponentiation display
        formula = MathTex("2^{n-1}", color=COLOR_FORMULA)
        # Fix issue 31: Relocate formula to area E5:F6 to represent the pattern for the whole sequence
        self.place_in_area(formula, 'E5', 'F6', scale_factor=1.2)
        
        # === ANIMATION ===

        # === Animation for Lecture Line 1 ===
        # "One point gives us a single region."
        self.play(self.lecture[0].animate.set_color(COLOR_VALUES))
        self.play(
            Create(h_line), 
            Create(v_line), 
            Write(header_n), 
            Write(header_regions)
        )
        self.play(Write(row_mobjects[0][0]), Write(row_mobjects[0][1]))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "Two points and one cut produce two regions."
        self.play(
            self.lecture[0].animate.set_color(COLOR_TEXT),
            self.lecture[1].animate.set_color(COLOR_VALUES)
        )
        self.play(Write(row_mobjects[1][0]), Write(row_mobjects[1][1]))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Three points with three cuts result in four regions."
        self.play(
            self.lecture[1].animate.set_color(COLOR_TEXT),
            self.lecture[2].animate.set_color(COLOR_VALUES)
        )
        self.play(Write(row_mobjects[2][0]), Write(row_mobjects[2][1]))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "Four points give eight, and five points give sixteen."
        self.play(
            self.lecture[2].animate.set_color(COLOR_TEXT),
            self.lecture[3].animate.set_color(COLOR_VALUES)
        )
        self.play(
            Write(row_mobjects[3][0]), Write(row_mobjects[3][1]),
            run_time=0.6
        )
        self.play(
            Write(row_mobjects[4][0]), Write(row_mobjects[4][1]),
            run_time=0.6
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "The sequence follows a perfect power of two pattern."
        self.play(
            self.lecture[3].animate.set_color(COLOR_TEXT),
            self.lecture[4].animate.set_color(COLOR_FORMULA)
        )
        self.play(Write(formula))
        self.play(Indicate(formula, color=COLOR_FORMULA))
        self.wait(2)
