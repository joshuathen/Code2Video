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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("Mathematical Mechanics (The Sliding Dot Product)", 
                          ["The operation follows mathematical dot product rules.", 
                           "We flip and shift the kernel across.", 
                           "This effectively averages nearby data samples."])
        
        # Create input 3x3 matrix
        input_data = [[1, 2, 0], [0, 1, 1], [2, 0, 1]]
        input_matrix = Matrix(input_data).set_color(WHITE)
        self.place_at_grid(input_matrix, 'B4', scale_factor=0.6)
        
        # Create kernel 3x3 matrix
        kernel_data = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        kernel_matrix = Matrix(kernel_data).set_color("#00FF00")
        self.place_at_grid(kernel_matrix, 'D2', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(input_matrix), FadeIn(kernel_matrix))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        # Show flip and shift (represented by a highlight move)
        self.lecture[1].set_color(GREEN)
        highlight = SurroundingRectangle(kernel_matrix, color=YELLOW, buff=0.1)
        self.play(Create(highlight))
        self.play(highlight.animate.move_to(input_matrix.get_center()))
        self.play(FadeOut(highlight))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        result_text = Text("Sum = 1*0 + 2*1 + 0*0 + 0*1 + 1*-4 + 1*1 + 2*0 + 0*1 + 1*0 = -1", 
                           font_size=20, color=YELLOW)
        self.place_at_grid(result_text, 'B5', scale_factor=0.7)
        self.play(Write(result_text))
        self.wait(2)
