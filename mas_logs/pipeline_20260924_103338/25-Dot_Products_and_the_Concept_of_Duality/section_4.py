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
    def construct(self):
        self.setup_layout("Visualization: The Dual Nature", [
            "Visualize the transformation by rotating the vector.",
            "Constant dot products create perpendicular level sets.",
            "These lines reveal the underlying dual landscape."
        ])

        # Define mobjects
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        vec = Vector([1, 1], color="#FFD700")
        line1 = Line(start=[-2, 0, 0], end=[2, 0, 0], color="#FFFFFF")
        line2 = Line(start=[0, -2, 0], end=[0, 2, 0], color="#FFFFFF")
        
        # Position them using area to avoid crowding
        self.place_in_area(grid_asset, 'B4', 'D6', scale_factor=0.8)
        self.place_in_area(vec, 'B4', 'D6', scale_factor=1.0)
        self.place_at_grid(line1, 'C5', scale_factor=0.7)
        self.place_at_grid(line2, 'C5', scale_factor=0.7)
        
        self.add(grid_asset)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFD700"))
        self.play(Create(vec), Create(line1), Create(line2))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00CED1"))
        # Rotation of the vector and level sets
        self.play(Rotate(vec, angle=PI/4, about_point=self.grid['C5']),
                  Rotate(line1, angle=PI/4, about_point=self.grid['C5']),
                  Rotate(line2, angle=PI/4, about_point=self.grid['C5']))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF5733"))
        # Highlighting the interaction
        self.play(Indicate(line1), Indicate(line2))
        self.wait(1)
