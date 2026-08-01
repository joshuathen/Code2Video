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
        # Title and Lecture Lines from Storyboard
        title_text = "The Bridge Between Worlds: Introduction"
        lecture_lines = [
            "Square matrices transform space within the same dimension.",
            "Non-square matrices move vectors between different dimensions.",
            "Matrix columns show where input basis vectors land."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Color constants from Storyboard
        COLOR_GRID = "#FFFFFF"
        COLOR_SHEAR = "#00FF00"
        COLOR_MATRIX = "#FFFF00"
        COLOR_3D = "#00FFFF"
        COLOR_BASIS = "#FF00FF"

        # === Animation for Lecture Line 1 ===
        # Animation: Show a standard 2D grid centered at the origin. (Color: #FFFFFF)
        self.lecture[0].set_color(COLOR_GRID)
        grid_2d = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": COLOR_GRID, "stroke_opacity": 0.5}
        )
        # Position grid in the middle-bottom area of the right side
        # Fix Issue 17: scale_factor set to 0.7 for consistency
        self.place_in_area(grid_2d, "C3", "F6", scale_factor=0.7)
        
        self.play(Create(grid_2d))
        self.wait(1)

        # Animation: Apply a 2x2 transformation, shearing the grid. (Color: #00FF00)
        # Update lecture line color to match
        self.play(self.lecture[0].animate.set_color(COLOR_SHEAR))
        
        matrix_shear = [[1, 1], [0, 1]]
        # Colored grid version for transformation
        grid_2d_target = grid_2d.copy().set_color(COLOR_SHEAR)
        
        self.play(
            Transform(grid_2d, grid_2d_target),
            grid_2d.animate.apply_matrix(matrix_shear),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Switch highlight to line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(COLOR_MATRIX)
        )

        # Animation: Display a 3x2 matrix next to the grid. (Color: #FFFF00)
        matrix_3x2 = Matrix(
            [["2", "1"], ["-1", "2"], ["0", "1"]],
            element_to_mobject_config={"color": COLOR_MATRIX}
        )
        matrix_3x2.get_brackets().set_color(COLOR_MATRIX)
        # Fix Issue 18: Move to A2 and scale to 0.8 for better space utilization
        self.place_at_grid(matrix_3x2, "A2", scale_factor=0.8)
        
        self.play(Write(matrix_3x2))
        self.wait(1)

        # Animation: Transition the 2D grid into a 3D plane. (Color: #00FFFF)
        # Update lecture line color to cyan
        self.play(self.lecture[1].animate.set_color(COLOR_3D))
        
        # Simulated 3D plane by rotating a NumberPlane in a 2D Scene
        plane_3d = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": COLOR_3D, "stroke_opacity": 0.6}
        )
        plane_3d.rotate(70 * DEGREES, axis=RIGHT)
        plane_3d.rotate(25 * DEGREES, axis=OUT)
        # Fix Issue 16: Position in C3-F6 for alignment with grid_2d
        self.place_in_area(plane_3d, "C3", "F6", scale_factor=0.7)

        self.play(
            FadeOut(grid_2d),
            FadeIn(plane_3d)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Switch highlight to line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(COLOR_BASIS)
        )

        # Animation: Highlight columns of the matrix as basis vectors in 3D. (Color: #FF00FF)
        # Use get_columns() - note this returns a list of VGroups
        cols_elements = matrix_3x2.get_columns()
        col1 = cols_elements[0]
        col2 = cols_elements[1]

        # Draw vectors on the tilted plane
        # Note: c2p on a rotated NumberPlane correctly maps coords to the transformed space.
        v_origin = plane_3d.c2p(0, 0)
        v1_end = plane_3d.c2p(1.5, -0.5)
        v2_end = plane_3d.c2p(0.8, 1.2)
        
        vec1 = Arrow(v_origin, v1_end, buff=0, color=COLOR_BASIS)
        vec2 = Arrow(v_origin, v2_end, buff=0, color=COLOR_BASIS)
        
        label1 = MathTex(r"\hat{i}'", color=COLOR_BASIS, font_size=24).next_to(vec1, RIGHT, buff=0.1)
        label2 = MathTex(r"\hat{j}'", color=COLOR_BASIS, font_size=24).next_to(vec2, UP, buff=0.1)

        self.play(
            col1.animate.set_color(COLOR_BASIS),
            Create(vec1),
            Write(label1)
        )
        self.wait(0.5)
        self.play(
            col2.animate.set_color(COLOR_BASIS),
            Create(vec2),
            Write(label2)
        )
        self.wait(3)
