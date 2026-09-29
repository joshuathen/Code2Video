from manim import *

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
            "Determinants measure the signed area of a parallelogram.",
            "Linear transformations scale the area of the unit square.",
            "This scaling factor reveals the transformation's underlying nature."
        ]
        self.setup_layout("Prerequisite Review: The Determinant and Area", lecture_lines)
        
        # Grid asset
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        # === Animation for Lecture Line 1 ===
        # Using grid asset and morphing NumberPlane
        grid = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], background_line_style={"stroke_opacity": 0.3})
        self.place_in_area(grid, 'C2', 'F6', scale_factor=0.5)
        self.place_in_area(grid_asset, 'C2', 'F6', scale_factor=0.2)
        self.play(FadeIn(grid), FadeIn(grid_asset))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Label 'Determinant' as the scaling factor of area.
        det_label = Tex("Determinant = Area Scaling", color="#FFD700")
        self.place_at_grid(det_label, 'B4', scale_factor=0.6)
        self.play(Write(det_label))
        self.lecture[1].set_color("#FFD700")

        # === Animation for Lecture Line 3 ===
        # Display a transformation matrix M changing the grid.
        matrix = MathTex(r"M = \begin{pmatrix} 2 & 1 \\ 0 & 1 \end{pmatrix}", color="#00FF00")
        self.place_at_grid(matrix, 'E5', scale_factor=0.7)
        self.play(Write(matrix))
        
        new_grid = grid.copy().apply_matrix([[2, 1], [0, 1]])
        self.play(Transform(grid, new_grid))
        self.lecture[2].set_color("#00FF00")
        
        self.wait(2)
