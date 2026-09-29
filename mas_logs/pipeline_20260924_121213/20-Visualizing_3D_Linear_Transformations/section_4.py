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
        lecture_lines = [
            "Rotations preserve lengths and volumes in 3D.",
            "Shear matrices create slide effects on grid layers.",
            "Scaling transforms change overall dimensions of objects."
        ]
        self.setup_layout("Application: Rotation & Shear", lecture_lines)
        
        # Elements
        cube_path = "/scratch/pawsey1357/jthen/Code2Video/assets/icon/cube.svg"
        cube = SVGMobject(cube_path, color=BLUE, fill_opacity=0.5)
        self.place_at_grid(cube, 'B5', scale_factor=0.8)
        
        rotation_label = Text("Rotation", font_size=20, color=BLUE)
        shear_label = Text("Shear", font_size=20, color=GREEN)
        scale_label = Text("Scale", font_size=20, color=RED)
        
        self.place_at_grid(rotation_label, 'C5', scale_factor=1.0)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Rotate(cube, angle=PI/4, about_point=self.grid['B5']))
        self.play(FadeIn(rotation_label))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(GREEN))
        shear_matrix = np.array([[1, 0.5], [0, 1]])
        self.play(ApplyMatrix(shear_matrix, cube), FadeOut(rotation_label))
        self.place_at_grid(shear_label, 'C5', scale_factor=1.0)
        self.play(FadeIn(shear_label))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(RED))
        self.play(cube.animate.scale(1.5), FadeOut(shear_label))
        self.place_at_grid(scale_label, 'C5', scale_factor=1.0)
        self.play(FadeIn(scale_label))
