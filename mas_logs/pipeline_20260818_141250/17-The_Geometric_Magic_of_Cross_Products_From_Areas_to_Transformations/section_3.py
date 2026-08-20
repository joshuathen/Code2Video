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
            "Scalar triple product calculates signed volume.",
            "Volume represents the 3D linear transformation.",
            "Parallelepiped volume is the determinant.",
            "Cross product is a 3D volume tool.",
            "Volume relates base area and height."
        ]
        self.setup_layout("Linear Transformations & Change of Volume", lecture_lines)
        
        # Initial shapes
        cube = Cube(side_length=1.5, fill_opacity=0.3, stroke_width=2)
        prism = Prism(dimensions=[1.5, 2.0, 1.2], fill_opacity=0.3, stroke_width=2)
        
        # === Animation for Lecture Line 1 ===
        # "Scalar triple product calculates signed volume."
        self.lecture[0].set_color("#FFFFFF")
        self.place_at_grid(cube, 'B3', scale_factor=0.9)
        self.play(Create(cube))

        # === Animation for Lecture Line 2 ===
        # "Volume represents the 3D linear transformation."
        self.lecture[1].set_color("#FFA500")
        self.play(Transform(cube, prism))

        # === Animation for Lecture Line 3 ===
        # "Parallelepiped volume is the determinant."
        self.lecture[2].set_color("#FF00FF")
        det_text = Text("det(A)", font_size=30).set_color("#FF00FF")
        self.place_at_grid(det_text, 'C3', scale_factor=0.8)
        self.play(Write(det_text))

        # === Animation for Lecture Line 4 ===
        # "Cross product is a 3D volume tool."
        self.lecture[3].set_color("#00FFFF")
        self.play(cube.animate.rotate(PI/4, axis=OUT), run_time=1.5)

        # === Animation for Lecture Line 5 ===
        # "Volume relates base area and height."
        self.lecture[4].set_color("#FFFFFF")
        vol_text = Text("Volume = |det(A)|", font_size=30).set_color("#FFFFFF")
        self.place_at_grid(vol_text, 'D3', scale_factor=0.8)
        self.play(FadeIn(vol_text))
        self.wait(2)
