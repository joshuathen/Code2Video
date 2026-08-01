from manim import *
import numpy as np

# Override Matrix to use Text instead of MathTex, bypassing the need for a LaTeX installation
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

class Matrix(VGroup):
    def __init__(self, matrix_data, color=WHITE, **kwargs):
        # Create elements using Text (Pango) which does not require LaTeX
        mob_matrix = VGroup(*[
            VGroup(*[Text(str(item), font_size=32) for item in row]).arrange(RIGHT, buff=0.6)
            for row in matrix_data
        ]).arrange(DOWN, buff=0.4)
        
        # Create brackets using Text mobjects
        l_bracket = Text("[", font_size=48).stretch_to_fit_height(mob_matrix.height + 0.2)
        l_bracket.next_to(mob_matrix, LEFT, buff=0.2)
        r_bracket = Text("]", font_size=48).stretch_to_fit_height(mob_matrix.height + 0.2)
        r_bracket.next_to(mob_matrix, RIGHT, buff=0.2)
        
        # Initialize the VGroup with the matrix elements and brackets
        super().__init__(mob_matrix, l_bracket, r_bracket, **kwargs)
        self.set_color(color)
        
        # Provide the 'elements' attribute for compatibility with the original Matrix class
        self.elements = VGroup(*[el for row in mob_matrix for el in row])

class Section1Scene(TeachingScene):
    def construct(self):
        # 1. Setup layout
        lecture_lines = [
            'A matrix is a function that moves space.', 
            'We track basis vectors i-hat and j-hat.', 
            '[Asset: Robo-Cat] stands on the initial unit square.', 
            'The matrix maps basis vectors to new positions.', 
            'This transformation stretches or rotates the entire space.'
        ]
        self.setup_layout("Prerequisite: The Matrix as a Function", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(LIGHT_GRAY)
        # Create a dark gray coordinate grid
        transformation_grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": "#333333", "stroke_opacity": 0.8}
        ).set_z_index(-1)
        
        # Issue 34 fix: Define the visual space for the transformation grid
        self.place_in_area(transformation_grid, 'A1', 'F6', scale_factor=1.0)
        grid_origin = transformation_grid.get_center()
        
        self.play(FadeIn(transformation_grid))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(RED)
        # Basis vectors: i-hat (red) and j-hat (green)
        i_hat = Vector([1, 0], color=RED).shift(grid_origin)
        j_hat = Vector([0, 1], color=GREEN).shift(grid_origin)
        
        i_hat_label = Text("i-hat", font_size=20, color=RED)
        j_hat_label = Text("j-hat", font_size=20, color=GREEN)
        
        # Issue 36 fix: Fix basis vector labels positioning
        self.place_at_grid(i_hat_label, 'E3', scale_factor=0.8)
        self.place_at_grid(j_hat_label, 'C2', scale_factor=0.8)
        
        self.play(GrowArrow(i_hat), Write(i_hat_label))
        self.play(GrowArrow(j_hat), Write(j_hat_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        # Issue 52/35 fix: Load and anchor Robo-Cat asset
        robo_cat = ImageMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/cat.png")
        self.place_at_grid(robo_cat, 'E2', scale_factor=0.5)
        
        self.play(FadeIn(robo_cat))
        
        # Move Robo-Cat onto the unit square bounded by the basis vectors
        unit_square_center = grid_origin + np.array([0.5, 0.5, 0])
        self.play(robo_cat.animate.move_to(unit_square_center).scale(1.5))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN)
        # Matrix A [[1, -1], [1, 1]] maps:
        # i (1,0) -> (1, 1)
        # j (0,1) -> (-1, 1)
        # Cat center (0.5, 0.5) -> (0, 1)
        matrix_vals = [[1, -1], [1, 1]]
        target_i_end = grid_origin + np.array([1, 1, 0])
        target_j_end = grid_origin + np.array([-1, 1, 0])
        target_cat_pos = grid_origin + np.array([0, 1, 0])
        
        # Warp the grid and objects smoothly
        self.play(
            transformation_grid.animate.apply_matrix(matrix_vals).move_to(grid_origin),
            i_hat.animate.put_start_and_end_on(grid_origin, target_i_end),
            j_hat.animate.put_start_and_end_on(grid_origin, target_j_end),
            robo_cat.animate.move_to(target_cat_pos).rotate(45 * DEGREES),
            i_hat_label.animate.move_to(target_i_end + 0.3 * UP),
            j_hat_label.animate.move_to(target_j_end + 0.3 * LEFT),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(YELLOW)
        # Display Matrix A with colored columns
        mat_a = Matrix(matrix_vals)
        self.place_at_grid(mat_a, 'A5', scale_factor=0.8)
        
        # Color columns: Col 1 Red (i-hat contribution), Col 2 Green (j-hat contribution)
        mat_a.elements[0].set_color(RED)
        mat_a.elements[2].set_color(RED)
        mat_a.elements[1].set_color(GREEN)
        mat_a.elements[3].set_color(GREEN)
        
        self.play(Write(mat_a))
        self.wait(2)
