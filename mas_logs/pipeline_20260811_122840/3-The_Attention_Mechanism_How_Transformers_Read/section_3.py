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
        self.setup_layout("Calculating Relevance: Dot Product & Softmax", [
            "Dot product measures item similarity.",
            "Softmax converts scores into probabilities.",
            "High scores highlight relevant words."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Dot product grid
        grid_data = [[8.5, 1.2, 0.5], [0.8, 7.2, 0.4], [0.3, 0.6, 9.1]]
        matrix = VGroup()
        for i in range(3):
            row = VGroup()
            for j in range(3):
                val = DecimalNumber(grid_data[i][j], num_decimal_places=1, font_size=30)
                box = Square(side_length=0.7, color=WHITE).add(val)
                row.add(box)
            matrix.add(row.arrange(RIGHT, buff=0))
        matrix.arrange(DOWN, buff=0)
        
        # Updated per issue: Line 66: self.place_in_area(matrix, 'A4', 'C6', scale_factor=0.6)
        self.place_in_area(matrix, "A4", "C6", scale_factor=0.6)
        self.play(FadeIn(matrix))
        self.lecture[0].set_color("#00FF00")

        # === Animation for Lecture Line 2 ===
        # Softmax visualization (labels)
        # Updated per issue: Line 73: self.place_at_grid(softmax_label, 'E5', scale_factor=0.8)
        softmax_label = Text("Softmax", font_size=24, color="#FF00FF")
        self.place_at_grid(softmax_label, "E5", scale_factor=0.8)
        self.play(Write(softmax_label))
        self.lecture[1].set_color("#FF00FF")

        # === Animation for Lecture Line 3 ===
        # Highlight heatmap
        highlights = VGroup(
            Rectangle(width=0.7, height=0.7, color="#FF0000", fill_opacity=0.3).move_to(matrix[0][0]),
            Rectangle(width=0.7, height=0.7, color="#FF0000", fill_opacity=0.3).move_to(matrix[1][1]),
            Rectangle(width=0.7, height=0.7, color="#FF0000", fill_opacity=0.3).move_to(matrix[2][2]),
        )
        self.play(FadeIn(highlights))
        self.lecture[2].set_color("#FF0000")
        self.wait(2)
