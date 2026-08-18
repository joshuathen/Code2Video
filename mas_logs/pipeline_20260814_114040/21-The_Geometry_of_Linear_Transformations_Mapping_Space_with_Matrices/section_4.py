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
        self.setup_layout("Dynamic Visualization: Rotation and Shear", [
            "Different matrices create distinct spatial effects easily.",
            "Rotation matrices turn the space around the origin.",
            "Shear matrices tilt the space while keeping lines parallel."
        ])

        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")

        # Grid and Unit vectors
        grid = NumberPlane(x_range=[-3, 3], y_range=[-3, 3], background_line_style={"stroke_opacity": 0.3})
        self.place_in_area(grid, 'B3', 'E5', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#0000FF")
        
        rot_label = Text("Rotation", color="#0000FF", font_size=20)
        self.place_at_grid(rot_label, 'A4', scale_factor=0.9)
        self.place_at_grid(compass, 'A5', scale_factor=0.4)
        
        # Rotating grid
        self.play(grid.animate.rotate(PI/2), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF7F00")
        
        shear_label = Text("Shear", color="#FF7F00", font_size=20)
        self.place_at_grid(shear_label, 'F4', scale_factor=0.9)
        
        # Shear transformation
        shear_matrix = [[1, 1], [0, 1]]
        self.play(grid.animate.apply_matrix(shear_matrix), run_time=2)
        self.wait(1)
        
        # Combined
        combined_label = Text("Combined", color=WHITE, font_size=20)
        self.place_at_grid(combined_label, 'F5', scale_factor=0.9)
        self.place_at_grid(protractor, 'F6', scale_factor=0.4)
        self.wait(2)
