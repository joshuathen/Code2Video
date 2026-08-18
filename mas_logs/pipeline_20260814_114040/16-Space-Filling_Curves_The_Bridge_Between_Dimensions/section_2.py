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
        self.setup_layout("The Peano Curve Construction", [
            "Divide the square into nine smaller squares.",
            "Connect the centers in a specific path.",
            "Iterate this process to refine the path."
        ])
        
        # Grid visual
        grid_square = Square(side_length=2.5, color=GRAY).set_stroke(width=1)
        self.place_in_area(grid_square, "B4", "E6", scale_factor=0.7)
        self.add(grid_square)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        lines = VGroup()
        for i in range(1, 3):
            # Drawing 3x3 grid using relative offsets
            h = Line(grid_square.get_corner(UL) + i*0.833*DOWN, grid_square.get_corner(UR) + i*0.833*DOWN)
            v = Line(grid_square.get_corner(UL) + i*0.833*RIGHT, grid_square.get_corner(DL) + i*0.833*RIGHT)
            lines.add(h, v)
        lines.set_color("#FFFFFF").set_stroke(width=1)
        self.play(Create(lines))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        centers = []
        for i in range(3):
            for j in range(3):
                centers.append(grid_square.get_corner(UL) + np.array([0.416 + j*0.833, -0.416 - i*0.833, 0]))
        path = VMobject(color="#00FF00", stroke_width=3)
        path.set_points_as_corners(centers)
        self.play(Create(path))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        self.play(FadeOut(path), FadeOut(lines))
        refined_path = VMobject(color="#FF00FF", stroke_width=2)
        refined_centers = []
        for i in range(9):
            for j in range(9):
                refined_centers.append(grid_square.get_corner(UL) + np.array([0.138 + j*0.277, -0.138 - i*0.277, 0]))
        refined_path.set_points_as_corners(refined_centers)
        self.play(Create(refined_path))
        self.wait(2)
