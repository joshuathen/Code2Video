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

class Section2Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Transformation Matrix", [
            "A 3x3 matrix represents linear transformations.",
            "Columns show destinations of basis vectors.",
            "Grid lines remain parallel and uniform."
        ])
        
        matrix = MathTex(
            r"M = \begin{pmatrix} x_1 & x_2 & x_3 \\ y_1 & y_2 & y_3 \\ z_1 & z_2 & z_3 \end{pmatrix}"
        )
        # Fix 24: Matrix position
        self.place_at_grid(matrix, 'C3', scale_factor=1.0)

        # Asset 17: SVG grid
        asset_grid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        self.play(Write(matrix))
        # Place asset 17
        self.place_at_grid(asset_grid, 'B5', scale_factor=0.3)
        self.play(FadeIn(asset_grid))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(BLUE))
        
        # Visualizing columns
        col1 = matrix[0][4:7]
        col2 = matrix[0][7:10]
        col3 = matrix[0][10:13]
        
        box1 = SurroundingRectangle(col1, color=RED)
        box2 = SurroundingRectangle(col2, color=GREEN)
        box3 = SurroundingRectangle(col3, color=BLUE)
        
        self.play(Create(box1))
        self.wait(0.5)
        self.play(Create(box2))
        self.wait(0.5)
        self.play(Create(box3))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(PURPLE))
        
        # Fix 25 & 26: Grid visual and grouping
        grid_plane = NumberPlane(x_range=[-2, 2], y_range=[-2, 2], axis_config={"stroke_opacity": 0.3})
        self.place_at_grid(grid_plane, 'E4', scale_factor=0.8)
        
        group_matrix_grid = VGroup(matrix, grid_plane)
        self.place_in_area(group_matrix_grid, 'B3', 'F4', scale_factor=0.9)
        
        self.play(Create(grid_plane))
        
        # Simple shear demonstration
        self.play(grid_plane.animate.apply_matrix([[1, 0.5], [0, 1]]))
        self.wait(2)
