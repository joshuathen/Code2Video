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
        lecture_lines = [
            "Matrices are concise transformation containers.",
            "Columns store new basis positions.",
            "Matrix multiplication applies these changes."
        ]
        self.setup_layout("The Matrix: Encoding the Transformation", lecture_lines)
        
        # Asset path
        container_asset = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/container.svg"
        
        # === Animation for Lecture Line 1 ===
        # Use container asset as frame
        container1 = SVGMobject(container_asset).scale(2.0)
        matrix = Matrix([[1, -1], [2, 1]]).set_color(WHITE)
        self.place_at_grid(container1, 'B3', scale_factor=1.0)
        self.place_at_grid(matrix, 'B3', scale_factor=0.8)
        self.play(FadeIn(container1), FadeIn(matrix))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(BLUE)
        
        # Highlight columns
        self.play(
            matrix.get_columns()[0].animate.set_color(RED),
            matrix.get_columns()[1].animate.set_color(GREEN)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(PURPLE)
        
        vec_v = Matrix([[3], [2]]).set_color(WHITE)
        self.place_at_grid(vec_v, 'B6', scale_factor=0.8)
        
        eq = MathTex("=").next_to(vec_v, LEFT)
        result = Matrix([[1], [8]]).set_color(WHITE)
        result.next_to(eq, LEFT)
        
        # container for multiplication
        container2 = SVGMobject(container_asset).scale(2.0)
        
        full_equation = VGroup(container2, matrix, eq, vec_v, result)
        self.place_in_area(full_equation, 'C3', 'D5', scale_factor=0.9)
        
        self.play(FadeIn(container2), FadeIn(vec_v), Write(eq), FadeIn(result))
        self.wait(2)
