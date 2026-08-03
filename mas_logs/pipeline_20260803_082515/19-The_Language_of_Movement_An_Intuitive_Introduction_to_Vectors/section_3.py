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
        title = "The Algebraic Vector: Coordinates"
        lines = [
            "We can describe vectors with numbers.",
            "The coordinate x represents horizontal movement.",
            "The coordinate y represents vertical movement.",
            "Together, [x, y] define the vector's shape.",
            "This allows computers to process movement."
        ]
        self.setup_layout(title, lines)

        # Colors
        VEC_COLOR = "#00FFFF"
        X_COLOR = "#FF8C00"
        Y_COLOR = "#32CD32"
        TEXT_COLOR = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(VEC_COLOR))
        
        # Vector v from E2 to A5 (3 units right, 4 units up)
        start_point = self.grid["E2"]
        end_point = self.grid["A5"]
        vector_v = Arrow(start=start_point, end=end_point, color=VEC_COLOR, buff=0, stroke_width=6)
        vector_label = MathTex("v", color=VEC_COLOR)
        self.place_at_grid(vector_label, "C3", scale_factor=0.8) # Between E2 and A5
        
        self.play(Create(vector_v), Write(vector_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(X_COLOR)
        )
        
        # Horizontal orange dashed line from E2 to E5
        horiz_line = DashedLine(
            start=self.grid["E2"], 
            end=self.grid["E5"], 
            color=X_COLOR,
            dash_length=0.1
        )
        horiz_label = Text("3 units Right", font_size=18, color=X_COLOR)
        # Fix: Issue 35: scale_factor=0.9
        self.place_in_area(horiz_label, 'F3', 'F4', scale_factor=0.9)
        
        self.play(Create(horiz_line), Write(horiz_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(Y_COLOR)
        )
        
        # Vertical lime dashed line from E5 to A5
        vert_line = DashedLine(
            start=self.grid["E5"], 
            end=self.grid["A5"], 
            color=Y_COLOR,
            dash_length=0.1
        )
        vert_label = Text("4 units Up", font_size=18, color=Y_COLOR)
        # Fix: Issue 33: use place_in_area for C6-D6
        self.place_in_area(vert_label, 'C6', 'D6', scale_factor=0.8)
        
        self.play(Create(vert_line), Write(vert_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(TEXT_COLOR)
        )
        
        # Coordinate pair [3, 4] next to arrow tip
        coord_pair = MathTex("[", "3", ",", "4", "]", color=TEXT_COLOR)
        # Fix: Issue 34: place_at_grid B6, scale_factor=1.0
        self.place_at_grid(coord_pair, 'B6', scale_factor=1.0)
        
        self.play(Write(coord_pair))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(TEXT_COLOR)
        )
        
        # Flash the numbers 3 and 4
        # horiz_line is 3, vert_line is 4
        self.play(
            Flash(horiz_line, color=X_COLOR, line_length=0.3),
            coord_pair[1].animate.set_color(X_COLOR),
            run_time=0.5
        )
        self.play(
            Flash(vert_line, color=Y_COLOR, line_length=0.3),
            coord_pair[3].animate.set_color(Y_COLOR),
            run_time=0.5
        )
        self.play(
            coord_pair[1].animate.set_color(TEXT_COLOR),
            coord_pair[3].animate.set_color(TEXT_COLOR)
        )
        self.wait(2)

        # Final cleanup for consistency
        self.play(self.lecture[4].animate.set_color(WHITE))
        self.wait(1)
