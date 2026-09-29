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
        self.setup_layout("Linear Transformations and Area-to-Volume", 
                          ["3D volumes are defined by three vectors.", 
                           "The scalar triple product computes this volume.", 
                           "It connects base area to final height."])
        
        # Elements
        # Using SVG asset
        cube = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg")
        matrix = MathTex(r"\\det(A) = \\mathbf{a} \\cdot (\\mathbf{b} \\times \\mathbf{c})", font_size=36)
        vol_label = Text("Volume", font_size=24)
        matrix_label = Text("T", font_size=24, color=WHITE)
        
        # === Animation for Lecture Line 1 ===
        # 3D volumes are defined by three vectors.
        self.place_at_grid(cube, "B3", scale_factor=0.6)
        cube.set_color("#87CEEB")
        self.play(FadeIn(cube))
        self.lecture[0].set_color("#87CEEB")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The scalar triple product computes this volume.
        self.place_in_area(matrix, 'B4', 'C6', scale_factor=0.9)
        self.play(Write(matrix))
        self.lecture[1].set_color("#32CD32")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # It connects base area to final height.
        self.place_at_grid(vol_label, 'D5', scale_factor=0.8)
        self.place_at_grid(matrix_label, 'B4', scale_factor=0.5)
        
        vol_label.set_color("#FFD700")
        matrix_label.set_color("#FFFFFF")
        
        group = VGroup(matrix, vol_label, matrix_label)
        self.place_in_area(group, 'B4', 'E6', scale_factor=0.85)
        
        self.play(FadeIn(vol_label), FadeIn(matrix_label))
        self.lecture[2].set_color("#FF4500")
        self.wait(1)
