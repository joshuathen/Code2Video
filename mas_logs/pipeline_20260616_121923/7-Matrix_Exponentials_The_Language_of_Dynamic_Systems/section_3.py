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

class Section3Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=32, color=WHITE).to_edge(UP, buff=0.5)
        self.add(self.title)

        # Left-side lecture content
        lecture_vgroup = VGroup(*[
            Text(line, font_size=22, color=WHITE)
            for line in lecture_lines
        ]).arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        
        lecture_vgroup.to_edge(LEFT, buff=1.0).shift(DOWN * 0.2)
        self.add(lecture_vgroup)

        # Define 6x6 grid for visual elements on the right side
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        # Positioning the grid on the right half of the screen
        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                x = 1.5 + j * 0.9
                y = 2.2 - i * 0.9
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def place_at_grid(self, mobject, grid_pos, scale_factor=0.6):
        mobject.scale(scale_factor)
        mobject.move_to(self.grid[grid_pos])
        return mobject

    def construct(self):
        # Define lecture content based on Matrix Exponentials context
        lecture_points = [
            "- Linear ODE Systems",
            "- Power Series Definition",
            "- Properties of exp(At)",
            "- Fundamental Solutions",
            "- State Transition Matrix"
        ]
        
        # Initialize Layout
        self.setup_layout("Matrix Exponentials: Dynamics", lecture_points)
        
        # Create visual elements without LaTeX dependencies
        matrix_elements = [
            [Text("0", font_size=24), Text("-1", font_size=24)],
            [Text("1", font_size=24), Text("0", font_size=24)]
        ]
        
        # Manual construction of matrix to avoid FileNotFoundError: 'latex'
        # MobjectMatrix internally uses MathTex for brackets, which triggers the error.
        matrix_grid = VGroup(
            VGroup(*matrix_elements[0]).arrange(RIGHT, buff=0.7),
            VGroup(*matrix_elements[1]).arrange(RIGHT, buff=0.7)
        ).arrange(DOWN, buff=0.7)
        
        # Create brackets using Text instead of MathTex
        l_bracket = Text("[", font_size=40).stretch_to_fit_height(matrix_grid.height + 0.2)
        l_bracket.next_to(matrix_grid, LEFT, buff=0.2)
        r_bracket = Text("]", font_size=40).stretch_to_fit_height(matrix_grid.height + 0.2)
        r_bracket.next_to(matrix_grid, RIGHT, buff=0.2)
        
        # Group components into matrix_a and mock the get_brackets method for compatibility
        matrix_a = VGroup(matrix_grid, l_bracket, r_bracket)
        matrix_a.get_brackets = lambda: VGroup(l_bracket, r_bracket)
        
        # Apply the desired color to the brackets
        matrix_a.get_brackets().set_color(BLUE_B)
        
        # Use Text instead of MathTex for formulas
        exp_formula = Text("exp(At) = sum (At)^n / n!", font_size=24, color=YELLOW)
        solution_eq = Text("x(t) = exp(At) x(0)", font_size=24, color=GREEN)
        
        # Place objects on the grid
        self.place_at_grid(matrix_a, "B2", scale_factor=0.8)
        self.place_at_grid(exp_formula, "D2", scale_factor=0.9)
        self.place_at_grid(solution_eq, "F2", scale_factor=0.9)
        
        # Animation sequence
        self.play(FadeIn(matrix_a, shift=RIGHT))
        self.wait(1)
        
        self.play(Write(exp_formula))
        self.wait(1)
        
        self.play(FadeIn(solution_eq, shift=UP))
        self.wait(2)
        
        # Demonstrate grid mobility
        self.play(
            matrix_a.animate.move_to(self.grid["B5"]),
            exp_formula.animate.move_to(self.grid["D5"]),
            solution_eq.animate.move_to(self.grid["F5"]),
            run_time=2,
            rate_func=smooth
        )
        
        self.wait(3)
