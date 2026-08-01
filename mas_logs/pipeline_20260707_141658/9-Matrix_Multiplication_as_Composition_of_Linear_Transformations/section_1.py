from manim import *
import numpy as np

# Use the provided TeachingScene base class without modification.
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
        # Data from storyboard
        title_text = "Prerequisite: The Matrix as a Machine"
        lecture_lines = [
            "A 2x2 matrix acts like a function on space.",
            "It transforms every point in the 2D grid.",
            "We represent this transformation as T(v) = Av."
        ]
        
        self.setup_layout(title_text, lecture_lines)

        # Colors
        COLOR_MATRIX = "#00FFFF"
        COLOR_FORMULA = "#FFFF00"
        COLOR_CAT = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Line 1: "A 2x2 matrix acts like a function on space."
        self.play(self.lecture[0].animate.set_color(COLOR_MATRIX))
        
        # 2D Grid (NumberPlane)
        # Fix from Issue 23: Reposition plane to avoid obstructing lecture lines and going off-screen
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4.0,
            y_length=4.0,
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_numbers": False} 
        )
        self.place_in_area(plane, 'C3', 'F6', scale_factor=0.7)
        
        # [Asset: /mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cat.png]
        # Fix from Issue 19: Use Asset from path provided
        cat = ImageMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cat.png")
        cat.height = 0.6
        
        # Initial position at (1,1) in plane coordinates
        cat.move_to(plane.coords_to_point(1, 1))
        
        self.play(Create(plane), FadeIn(cat))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: "It transforms every point in the 2D grid."
        self.play(self.lecture[1].animate.set_color(COLOR_MATRIX))
        
        # Matrix A = [[0, -1], [1, 0]]
        # Fix from Issue 24: Reposition matrix_a_tex to 'A4' to avoid overlap
        matrix_a_tex = Text(
            "A = [[0, -1], [1, 0]]", 
            color=COLOR_MATRIX,
            font_size=24
        )
        self.place_at_grid(matrix_a_tex, 'A4', scale_factor=0.8)
        
        # Formula T(v) = Av
        # Fix from Issue 25: Reposition formula_tex to 'A6' to avoid obstruction
        formula_tex = Text(
            "T(v) = Av", 
            color=COLOR_FORMULA,
            font_size=24
        )
        self.place_at_grid(formula_tex, 'A6', scale_factor=0.8)

        self.play(Write(matrix_a_tex), Write(formula_tex))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: "We represent this transformation as T(v) = Av."
        self.play(self.lecture[2].animate.set_color(COLOR_FORMULA))
        
        # Matrix to apply (90 degree counter-clockwise rotation)
        matrix = [[0, -1], [1, 0]]
        
        # Transform the space and the object
        # Use about_point=plane's origin to keep the transformation local to the grid area
        origin = plane.coords_to_point(0, 0)
        
        self.play(
            plane.animate.apply_matrix(matrix, about_point=origin),
            cat.animate.apply_matrix(matrix, about_point=origin),
            run_time=3
        )
        self.wait(2)
