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
        # 1. Setup Layout
        title_text = "Prerequisite Check: Matrices as Functions"
        lecture_lines = [
            "Matrix multiplication transforms vectors into new ones.",
            "Think of it as warping the underlying grid.",
            "A vector moves to a new destination."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Assets
        grid_asset_path = "/mmfs1/data/home/jthen/Code2Video/assets/icon/grid.svg"
        
        # === Animation for Lecture Line 1 ===
        # Highlight Line 1
        self.play(self.lecture[0].animate.set_color(YELLOW), run_time=0.5)

        # Show unit grid [Asset]
        grid_system = SVGMobject(grid_asset_path)
        # Issue 35: Anchor grid system
        self.place_in_area(grid_system, 'A1', 'F6', scale_factor=2.0)
        grid_system.set_stroke(opacity=0.3)
        
        # Basis vectors i (#FF0000) and j (#00FF00)
        # Origin relative to grid_system's center
        origin = grid_system.get_center()
        unit_len = 1.0 # Approximate unit length relative to SVG scale
        
        basis_i = Arrow(origin, origin + RIGHT * unit_len, color="#FF0000", buff=0)
        basis_j = Arrow(origin, origin + UP * unit_len, color="#00FF00", buff=0)
        label_i = Text("i", font_size=18, color="#FF0000").next_to(basis_i, DOWN, buff=0.1)
        label_j = Text("j", font_size=18, color="#00FF00").next_to(basis_j, LEFT, buff=0.1)

        self.play(Create(grid_system), run_time=1)
        self.play(GrowArrow(basis_i), GrowArrow(basis_j), Write(label_i), Write(label_j))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight Line 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(BLUE),
            run_time=0.5
        )

        # Animate the grid warping into a skewed state using a generic matrix A
        # A = [[1.5, 0.5], [0.2, 1.0]]
        matrix_a = [[1.5, 0.5, 0], [0.2, 1.0, 0], [0, 0, 1]]
        
        # We transform the grid and basis vectors together
        transformation_group = VGroup(grid_system, basis_i, basis_j, label_i, label_j)
        
        # Custom transformation logic for the group
        self.play(
            transformation_group.animate.apply_matrix(matrix_a, about_point=origin),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight Line 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FFFF"),
            run_time=0.5
        )

        # Trace a vector v (#00FFFF) moving to its transformed position w = Av
        # Initial vector v (pre-transformation state would be [1, 1])
        # Since the grid is already transformed, we show v moving from its "pre-image" 
        # position relative to skewed basis to its final position or simply show the transformation.
        
        # Define v in original coordinates
        v_start_coords = np.array([1, 1, 0])
        v_end_coords = np.dot(np.array([[1.5, 0.5, 0], [0.2, 1.0, 0], [0, 0, 1]]), v_start_coords)
        
        v_vector = Arrow(origin, origin + (RIGHT * v_start_coords[0] + UP * v_start_coords[1]) * unit_len, color="#00FFFF", buff=0)
        v_label = Text("v", font_size=20, color="#00FFFF")
        # Issue 36: Anchor 'v' label
        self.place_at_grid(v_label, 'C4', scale_factor=0.8)

        self.play(GrowArrow(v_vector), FadeIn(v_label))
        self.wait(0.5)

        # Transform v to w = Av
        w_vector_target = Arrow(origin, origin + (RIGHT * v_end_coords[0] + UP * v_end_coords[1]) * unit_len, color="#00FFFF", buff=0)
        w_label = Text("Av", font_size=20, color="#CF50AC") # Pink/Purple for transformed state as per previous context
        # Issue 37: Anchor 'Av' label
        self.place_at_grid(w_label, 'B6', scale_factor=0.8)

        self.play(
            Transform(v_vector, w_vector_target),
            Transform(v_label, w_label),
            run_time=1.5
        )
        self.wait(2)
