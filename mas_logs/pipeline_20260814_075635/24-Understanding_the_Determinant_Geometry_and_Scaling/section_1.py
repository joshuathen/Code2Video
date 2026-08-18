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
            "Determinants define the scaling factor of transformations.",
            "Matrices stretch or squish space geometry.",
            "The determinant measures area change after transformation."
        ]
        self.setup_layout("Intuitive Hook: The Scaling Factor", lecture_lines)
        
        # Grid Setup
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_asset, 'C2', 'E5', scale_factor=0.8)
        
        # i-hat, j-hat vectors
        i_hat = Vector(RIGHT, color=WHITE).shift(grid_asset.get_center())
        j_hat = Vector(UP, color=WHITE).shift(grid_asset.get_center())
        
        square = Square(side_length=0.5, fill_opacity=0.3, color=WHITE)
        square.move_to(grid_asset.get_center() + np.array([0.25, 0.25, 0]))
        
        # === Animation for Lecture Line 1 ===
        # Determinants define the scaling factor of transformations.
        self.play(FadeIn(grid_asset), GrowArrow(i_hat), GrowArrow(j_hat), Create(square))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Matrices stretch or squish space geometry.
        self.lecture[1].set_color("#FF00FF")
        matrix = [[1.5, 0.5], [0, 1.5]]
        
        def apply_matrix(pt):
            res = np.dot(matrix, pt)
            return res
            
        new_square = Polygon(
            grid_asset.get_center() + np.array(list(apply_matrix([0,0])) + [0]),
            grid_asset.get_center() + np.array(list(apply_matrix([0.5,0])) + [0]),
            grid_asset.get_center() + np.array(list(apply_matrix([0.5,0.5])) + [0]),
            grid_asset.get_center() + np.array(list(apply_matrix([0,0.5])) + [0]),
            fill_opacity=0.3, color="#FF00FF"
        )
        
        self.play(Transform(square, new_square))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The determinant measures area change after transformation.
        self.lecture[2].set_color("#FFFF00")
        det_label = Text("det(M) = 2.25", font_size=20, color="#FFFF00")
        
        # Fix 20: det_label at B6
        self.place_at_grid(det_label, 'B6', scale_factor=0.9)
        
        # Fix 21: Bounding box area C4-E6
        transformation_box = SurroundingRectangle(square, color="#00FFFF", buff=0.1)
        self.place_in_area(transformation_box, 'C4', 'E6', scale_factor=0.85)
        
        # Fix 22: Lecture alignment A1-C1 (re-apply scale for consistency)
        # Note: self.lecture is a VGroup. 
        # The prompt instructed replacing with code that implements these fixes.
        # However, the lecture is defined in setup_layout. 
        # To respect the constraint: avoid altering setup_layout if possible.
        # The critique suggests moving the lecture group itself.
        self.place_in_area(self.lecture, 'A1', 'C1', scale_factor=0.95)
        
        self.play(FadeIn(det_label), Create(transformation_box))
        self.play(transformation_box.animate.scale(1.1), run_time=0.5)
        self.play(transformation_box.animate.scale(1/1.1), run_time=0.5)
        self.wait(2)
