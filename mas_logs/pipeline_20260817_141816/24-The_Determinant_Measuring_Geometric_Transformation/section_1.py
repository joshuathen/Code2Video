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
        self.setup_layout("Intuitive Hook: The Scaling Factor", [
            "Matrices represent a transformation of space.",
            "Think of space as a cat's territory.",
            "Transformations stretch, rotate, or shear space.",
            "The determinant acts as a scaling factor.",
            "It measures how area changes after transformation."
        ])

        # Assets
        cat_icon = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cat.png")

        # Animation objects
        unit_square = Square(side_length=1.0, color="#FF00FF", fill_opacity=0.5)
        # Apply fix 22
        self.place_at_grid(unit_square, 'C4', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        # Integrate cat icon with unit square
        cat_icon_1 = cat_icon.copy()
        self.place_at_grid(cat_icon_1, 'C5', scale_factor=0.2)
        self.play(FadeIn(unit_square), FadeIn(cat_icon_1))
        self.lecture[0].set_color("#FF00FF")

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        grid_copy = unit_square.copy()
        self.play(grid_copy.animate.scale(2).set_color("#FFFF00"))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#00FFFF"))
        # Represent shear using apply_matrix
        self.play(grid_copy.animate.apply_matrix([[1, 0.5], [0, 1]]))

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#00FFFF"))
        matrix_text = MathTex(r"det(A) = 2", color="#00FFFF")
        # Apply fix 21
        self.place_at_grid(matrix_text, 'E5', scale_factor=0.9)
        self.play(Write(matrix_text))

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF0000"))
        final_rect = Rectangle(width=2, height=1, color="#FF0000", fill_opacity=0.3)
        # Apply fix 20
        self.place_at_grid(final_rect, 'B5', scale_factor=0.8)
        
        # Integrate cat icon again
        cat_icon_2 = cat_icon.copy()
        self.place_at_grid(cat_icon_2, 'B6', scale_factor=0.2)
        
        self.play(FadeIn(final_rect), FadeIn(cat_icon_2))
        self.play(Indicate(final_rect))
