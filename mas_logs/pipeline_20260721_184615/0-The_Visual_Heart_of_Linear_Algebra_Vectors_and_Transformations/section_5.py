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

class Section5Scene(TeachingScene):
    def construct(self):
        title = "The Determinant: Scaling Area"
        lines = [
            "The determinant measures how transformations scale area.",
            "A determinant of two means every area doubles.",
            "If the value is zero, space squishes completely.",
            "This signifies a loss of a dimension into lines.",
            "Geometric intuition reveals the hidden power of determinants."
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_SQUARE = "#FFFF00"
        COLOR_DET = "#FF00FF"
        COLOR_IHAT = "#FF4444"
        COLOR_JHAT = "#44FF44"
        COLOR_TEXT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # self.wait(1.5). Create a 1x1 square unit (#FFFF00) using i-hat and j-hat, labeled 'Area = 1'.
        self.lecture[0].set_color(COLOR_SQUARE)
        
        # Grid/Coordinate system container
        plane = NumberPlane(
            x_range=[-2, 4], y_range=[-2, 3],
            x_length=4, y_length=4,
            background_line_style={"stroke_opacity": 0.4}
        )
        self.place_in_area(plane, 'A3', 'F6')
        
        # 1x1 Square
        unit_square = Square(side_length=plane.get_x_unit_size(), fill_opacity=0.3, fill_color=COLOR_SQUARE, stroke_color=COLOR_SQUARE)
        unit_square.move_to(plane.c2p(0.5, 0.5))
        
        # Vectors
        ihat = Arrow(plane.c2p(0, 0), plane.c2p(1, 0), buff=0, color=COLOR_IHAT, stroke_width=4)
        jhat = Arrow(plane.c2p(0, 0), plane.c2p(0, 1), buff=0, color=COLOR_JHAT, stroke_width=4)
        
        area_label = Text("Area = 1", font_size=20, color=COLOR_TEXT)
        self.place_at_grid(area_label, 'B2', scale_factor=0.8)

        self.play(Create(plane), FadeIn(unit_square), GrowArrow(ihat), GrowArrow(jhat), Write(area_label))
        self.wait(1.5)

        # === Animation for Lecture Line 2 ===
        # self.wait(2.0). Transform the grid to scale the area by 2.0, updating the square and its label.
        self.lecture[0].set_color(COLOR_TEXT)
        self.lecture[1].set_color(COLOR_SQUARE)
        
        # Target matrix for scaling: [[2, 0], [0, 1]] -> Area doubles
        matrix_scale = [[2, 0], [0, 1]]
        
        # New square (Rectangle now)
        scaled_square = Rectangle(
            width=plane.get_x_unit_size() * 2, 
            height=plane.get_y_unit_size(), 
            fill_opacity=0.3, 
            fill_color=COLOR_SQUARE, 
            stroke_color=COLOR_SQUARE
        )
        scaled_square.move_to(plane.c2p(1, 0.5))
        
        new_area_label = Text("Area = 2", font_size=20, color=COLOR_TEXT)
        self.place_at_grid(new_area_label, 'B2', scale_factor=0.8)

        self.play(
            plane.animate.apply_matrix(matrix_scale),
            Transform(unit_square, scaled_square),
            ihat.animate.put_start_and_end_on(plane.c2p(0,0), plane.c2p(2, 0)),
            jhat.animate.put_start_and_end_on(plane.c2p(0,0), plane.c2p(0, 1)),
            Transform(area_label, new_area_label),
            run_time=2
        )
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # self.wait(1.5). Display 'det = 2' in magenta (#FF00FF) and flash the scaled area.
        self.lecture[1].set_color(COLOR_TEXT)
        self.lecture[2].set_color(COLOR_DET)
        
        det_label = Text("det = 2", font_size=24, color=COLOR_DET)
        self.place_at_grid(det_label, 'A2', scale_factor=0.8)
        
        self.play(FadeIn(det_label))
        self.play(Indicate(unit_square, color=COLOR_DET))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        # self.wait(2.0). Morph the grid into a line (det = 0) and show the square flattening.
        self.lecture[2].set_color(COLOR_TEXT)
        self.lecture[3].set_color(COLOR_DET)
        
        # Matrix [[1, 1], [1, 1]] -> det = 0 (space squashes to line y=x)
        matrix_squash = [[1, 1], [1, 1]]
        
        # For a squashed square, it becomes a line segment from (0,0) to (2,2)
        flat_square = Line(plane.c2p(0,0), plane.c2p(2,2), color=COLOR_SQUARE, stroke_width=6)
        
        new_det_label = Text("det = 0", font_size=24, color=COLOR_DET)
        self.place_at_grid(new_det_label, 'A2', scale_factor=0.8)
        
        self.play(
            plane.animate.apply_matrix(matrix_squash),
            Transform(unit_square, flat_square),
            ihat.animate.put_start_and_end_on(plane.c2p(0,0), plane.c2p(1,1)),
            jhat.animate.put_start_and_end_on(plane.c2p(0,0), plane.c2p(1,1)),
            Transform(det_label, new_det_label),
            run_time=2
        )
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # self.wait(1.5). Fade in label 'Area = 0' and 'det = 0' (#FF00FF) to conclude scaling concept.
        self.lecture[3].set_color(COLOR_TEXT)
        self.lecture[4].set_color(COLOR_DET)
        
        zero_area_label = Text("Area = 0", font_size=20, color=COLOR_TEXT)
        self.place_at_grid(zero_area_label, 'B2', scale_factor=0.8)
        
        self.play(Transform(area_label, zero_area_label))
        self.play(Indicate(det_label, color=COLOR_DET))
        self.wait(1.5)
