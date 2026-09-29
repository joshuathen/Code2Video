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

class Section1Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "The determinant measures scaling of area.",
            "Start with a unit square area.",
            "Apply a matrix transformation now.",
            "Observe the square become a parallelogram.",
            "The change in area is determinant."
        ]
        self.setup_layout("The Determinant: Scaling Factor", lecture_lines)
        
        # Define objects
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg
        icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        square = Square(side_length=2, color="#FF5733", fill_opacity=0.3)
        self.place_at_grid(square, 'D4', scale_factor=0.8)
        self.place_at_grid(icon, 'B6', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF5733"))
        self.play(Create(square))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#33FF57"))
        parallelogram = Polygon(
            np.array([-1, -1, 0]), np.array([1, -1, 0]), 
            np.array([2, 1, 0]), np.array([0, 1, 0]),
            color=BLUE, fill_opacity=0.3
        )
        # Using place_in_area as recommended by issue 27, adjusting for D4/E5 constraint
        self.place_in_area(parallelogram, 'C4', 'E5', scale_factor=0.75)
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#3357FF"))
        self.play(Transform(square, parallelogram))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF33FF"))
        # Using icon to finalize
        self.play(FadeIn(icon))
        self.wait(2)
