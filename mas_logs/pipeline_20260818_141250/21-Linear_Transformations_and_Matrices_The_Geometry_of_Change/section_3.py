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

class Section3Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Matrices track where basis vectors i-hat and j-hat land.",
            "Columns define the destination of basis vectors.",
            "All points transform based on these new basis positions.",
            "Visual demonstration of basis movement.",
            "Linear combinations map the entire space."
        ]
        self.setup_layout("The Matrix as a Transformation Recipe", lecture_lines)
        
        # --- Assets ---
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        vector_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/vector.svg")
        coord_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/coordinate.svg")
        
        # --- Prepare Visuals ---
        # Matrix M = [[2, 0], [0, 1]]
        matrix = MathTex(r"M = \begin{bmatrix} 2 & 0 \\ 0 & 1 \end{bmatrix}", font_size=36)
        self.place_at_grid(matrix, 'B4', scale_factor=1.0)
        
        # Basis vectors
        i_vec = Vector(RIGHT, color=RED)
        j_vec = Vector(UP, color=GREEN)
        basis_group = VGroup(i_vec, j_vec)
        self.place_in_area(basis_group, 'D4', 'F6', scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        # Display matrix with grid asset
        self.place_at_grid(grid_asset, 'C2', scale_factor=0.5)
        self.play(FadeIn(matrix), FadeIn(grid_asset))
        self.lecture[0].set_color("#FFFFFF")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(basis_group))
        self.lecture[1].set_color("#00FF00")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # i moves to (2,0)
        i_target = Vector([2, 0], color=RED)
        self.place_at_grid(i_target, 'D5', scale_factor=0.7)
        self.play(Transform(i_vec, i_target))
        self.lecture[2].set_color("#FF0000")
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.place_at_grid(vector_asset, 'E5', scale_factor=0.4)
        self.play(FadeIn(vector_asset))
        self.lecture[3].set_color("#FFFF00")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.place_at_grid(coord_asset, 'B6', scale_factor=0.5)
        self.play(FadeIn(coord_asset))
        self.lecture[4].set_color("#00FFFF")
        self.wait(2)
