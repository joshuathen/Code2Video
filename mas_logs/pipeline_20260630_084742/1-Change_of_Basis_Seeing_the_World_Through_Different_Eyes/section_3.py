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
        # Setup title and lecture lines
        title_text = "The Bridge: The Transition Matrix (P)"
        lecture_lines = [
            "We need a translator between Bob and Z-4.",
            "The transition matrix P acts as our dictionary.",
            "Each column describes Z-4's vectors in Bob's view.",
            "For Z-4, P encodes his diagonal basis vectors.",
            "This matrix maps Z-4's coordinates back to Bob's."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Coordinate Plane (Bob's grid)
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            axis_config={"stroke_width": 2},
            background_line_style={"stroke_opacity": 0.3}
        )
        # Resolving Issue 35 & 47: Refine coordinate plane scaling/positioning
        self.place_in_area(plane, 'B3', 'D5', scale_factor=1.0)

        origin = plane.get_origin()

        # Z-4's basis vectors
        # Diagonal vectors B1 = (1, 1) and B2 = (-1, 1)
        b1_vec = Vector(plane.coords_to_point(1, 1) - origin, color="#00FF00").shift(origin)
        b2_vec = Vector(plane.coords_to_point(-1, 1) - origin, color="#FFFF00").shift(origin)

        # Labels for basis vectors
        b1_label = Text("B1 (1,1)", font_size=18, color="#00FF00")
        b1_label.next_to(b1_vec.get_end(), UR, buff=0.1)
        
        b2_label = Text("B2 (-1,1)", font_size=18, color="#FFFF00")
        b2_label.next_to(b2_vec.get_end(), UL, buff=0.1)

        # Matrix P Construction (Avoiding LaTeX/MathTex)
        # Column 1: [1, 1]^T (Z-4's B1 in Bob's basis)
        col1 = VGroup(
            Text("1", font_size=24, color="#00FF00"),
            Text("1", font_size=24, color="#00FF00")
        ).arrange(DOWN, buff=0.2)
        
        # Column 2: [-1, 1]^T (Z-4's B2 in Bob's basis)
        col2 = VGroup(
            Text("-1", font_size=24, color="#FFFF00"),
            Text("1", font_size=24, color="#FFFF00")
        ).arrange(DOWN, buff=0.2)

        matrix_vals = VGroup(col1, col2).arrange(RIGHT, buff=0.4)
        
        # Stylized Matrix Brackets
        l_bracket = Text("[", font_size=40, color=WHITE)
        r_bracket = Text("]", font_size=40, color=WHITE)
        l_bracket.next_to(matrix_vals, LEFT, buff=0.1)
        r_bracket.next_to(matrix_vals, RIGHT, buff=0.1)
        
        matrix = VGroup(l_bracket, matrix_vals, r_bracket)
        # Resolving Issue 34 & 47: Refine matrix centering and scale
        self.place_in_area(matrix, 'F3', 'F5', scale_factor=0.8)

        # Matrix Label
        matrix_label = Text("Transition Matrix P", font_size=22, color=WHITE)
        self.place_in_area(matrix_label, 'E3', 'E5', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Create(plane), run_time=1)
        self.play(GrowArrow(b1_vec), GrowArrow(b2_vec))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(YELLOW))
        self.play(Write(b1_label), Write(b2_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(YELLOW))
        # Visual construction of matrix P
        self.play(Write(l_bracket), Write(r_bracket))
        self.play(Write(col1), Write(col2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(YELLOW))
        # Emphasize that columns are the diagonal basis vectors
        self.play(Indicate(col1), Indicate(col2))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(YELLOW))
        self.play(Write(matrix_label))
        self.wait(2)
