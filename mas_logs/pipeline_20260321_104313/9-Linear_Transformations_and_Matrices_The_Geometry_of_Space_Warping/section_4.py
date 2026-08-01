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
    def create_text_matrix(self, matrix_data, bracket_color=RED):
        """
        Creates a matrix-like mobject using Text instead of MathTex to avoid 
        LaTeX dependencies.
        """
        # Create elements using Text
        elements_group = VGroup(*[
            VGroup(*[Text(str(item), font_size=24) for item in row])
            for row in matrix_data
        ])
        
        # Arrange elements in grid
        for row in elements_group:
            row.arrange(RIGHT, buff=0.6)
        elements_group.arrange(DOWN, buff=0.4)
        
        # Create brackets manually using stretched Text objects
        l_brack = Text("[", font_size=48).stretch_to_fit_height(elements_group.height + 0.2)
        l_brack.next_to(elements_group, LEFT, buff=0.1)
        r_brack = Text("]", font_size=48).stretch_to_fit_height(elements_group.height + 0.2)
        r_brack.next_to(elements_group, RIGHT, buff=0.1)
        
        l_brack.set_color(bracket_color)
        r_brack.set_color(bracket_color)
        
        matrix_mob = VGroup(elements_group, l_brack, r_brack)
        
        # Mimic standard Matrix methods
        matrix_mob.get_columns = lambda: VGroup(*[
            VGroup(*[elements_group[r][c] for r in range(len(elements_group))]) 
            for c in range(len(elements_group[0]))
        ])
        matrix_mob.get_brackets = lambda: VGroup(l_brack, r_brack)
        
        return matrix_mob

    def construct(self):
        # MANDATORY: setup_layout with specified lines
        lecture_lines = [
            "A matrix stores these landing spots in its columns.",
            "Stretching the grid updates the matrix with new values.",
            "Columns represent where basis vectors end up after transformation."
        ]
        self.setup_layout("Matrices: The Coordinate Containers", lecture_lines)

        # Color definitions
        color_1, color_2, color_3 = YELLOW, RED, GREEN

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(color_1))
        
        # Initial identity matrix
        matrix = self.create_text_matrix([[1, 0], [0, 1]])
        # Fix 42: Initial matrix positioning occupies too much vertical space (A2-B5). 
        # New area: 'A2' to 'C4', scale factor 0.7
        self.place_in_area(matrix, "A2", "C4", scale_factor=0.7)
        self.play(FadeIn(matrix))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(color_2))
        
        # Coordinate grid setup
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_opacity": 0.4}
        )
        i_vec = Vector([1, 0], color=RED)
        j_vec = Vector([0, 1], color=GREEN)
        grid_group = VGroup(plane, i_vec, j_vec)
        
        # Fix 41: Coordinate grid is scaled too small (0.4) and too close to bottom edge.
        # New area: 'D2' to 'F6', scale factor 0.7
        self.place_in_area(grid_group, "D2", "F6", scale_factor=0.7)
        self.play(FadeIn(grid_group))
        
        # Apply scaling transformation [3x horizontal, 2x vertical]
        self.play(
            grid_group.animate.apply_matrix([[3, 0], [0, 2]]),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(color_3))
        
        # Updated matrix representing the scaling transformation
        new_matrix = self.create_text_matrix([[3, 0], [0, 2]])
        # Fix 40: Labels overlap significantly beneath the matrix.
        # New area: 'A2' to 'C5', scale factor 0.6 to provide more space
        self.place_in_area(new_matrix, "A2", "C5", scale_factor=0.6)
        
        # Column destination labels
        label_i = Text("i-hat destination", font_size=14, color=RED)
        label_j = Text("j-hat destination", font_size=14, color=GREEN)
        
        # Position labels within 1 grid unit of the matrix columns
        label_i.next_to(new_matrix.get_columns()[0], DOWN, buff=0.2)
        label_j.next_to(new_matrix.get_columns()[1], DOWN, buff=0.2)
        
        self.play(
            ReplacementTransform(matrix, new_matrix),
            FadeIn(label_i),
            FadeIn(label_j)
        )
        self.wait(2)
