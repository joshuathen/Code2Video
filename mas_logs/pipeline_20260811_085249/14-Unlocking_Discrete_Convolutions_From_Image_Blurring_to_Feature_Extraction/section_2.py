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
        self.setup_layout("The Mechanics: Weighted Summation", [
            "Overlay the kernel onto the input grid.",
            "Multiply corresponding values element-wise.",
            "Sum the results to get a single output."
        ])
        
        # Asset placeholders (none.svg does not exist, so creating a representative visual)
        # As instructed: use the provided elements in Animation Description. 
        # Since [Asset: .../none.svg] is effectively empty, use a placeholder.
        input_asset = Square(side_length=0.5, color=WHITE, fill_opacity=0.3)
        output_asset = Circle(radius=0.3, color="#FF4500", fill_opacity=0.3)

        # Create 3x3 input matrix
        input_matrix = VGroup(*[Square(side_length=0.8, color=WHITE) for _ in range(9)])
        input_matrix.arrange_in_grid(3, 3, buff=0)
        # Shifted to avoid overlap and lecture space
        self.place_in_area(input_matrix, "A3", "B4", scale_factor=0.5)
        
        # Create 3x3 kernel matrix
        kernel_matrix = VGroup(*[Square(side_length=0.8, color="#00FFFF") for _ in range(9)])
        kernel_matrix.arrange_in_grid(3, 3, buff=0)
        self.place_in_area(kernel_matrix, "A5", "B6", scale_factor=0.5)
        
        self.play(Create(input_matrix), FadeIn(input_asset.next_to(input_matrix, UP)))
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FFFF")
        self.play(Create(kernel_matrix))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        # Highlighting pairs
        highlights = VGroup(*[SurroundingRectangle(m, color="#FF00FF", buff=0) for m in input_matrix])
        self.play(Create(highlights))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF4500")
        output_pixel = Square(side_length=0.8, color="#FF4500")
        # Shifted up to D5 to avoid lower grid issues
        self.place_at_grid(output_pixel, "D5", scale_factor=0.6)
        self.play(FadeIn(output_pixel), FadeIn(output_asset.next_to(output_pixel, DOWN)))
        self.wait(2)
