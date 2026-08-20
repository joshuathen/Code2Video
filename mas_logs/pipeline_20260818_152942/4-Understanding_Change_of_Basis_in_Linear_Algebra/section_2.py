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
        self.setup_layout("Defining the Transformation Matrix", [
            "We relate two distinct bases.",
            "Basis vectors map to coordinates.",
            "These form the transformation matrix.",
            "Matrix P bridges our views.",
            "Conversion relies on matrix multiplication."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Define basis vectors i and j for System A (#FF0000)
        i_vec = Arrow(ORIGIN, RIGHT, color="#FF0000")
        j_vec = Arrow(ORIGIN, UP, color="#FF0000")
        vec_group = VGroup(i_vec, j_vec)
        self.place_in_area(vec_group, 'B2', 'C3', scale_factor=0.9)
        self.play(Create(i_vec), Create(j_vec))
        self.lecture[0].set_color("#FF0000")

        # === Animation for Lecture Line 2 ===
        # Show how i and j map to new positions in B
        i_new = Arrow(ORIGIN, RIGHT + 0.5 * UP, color="#00FF00")
        j_new = Arrow(ORIGIN, -0.5 * RIGHT + UP, color="#00FF00")
        self.place_at_grid(i_new, 'B5', scale_factor=0.8)
        self.place_at_grid(j_new, 'C5', scale_factor=0.8)
        self.play(ReplacementTransform(i_vec.copy(), i_new), ReplacementTransform(j_vec.copy(), j_new))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # Construct a 2x2 grid representing Matrix M
        # Using SVGMobject for /scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg
        grid_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        matrix_m = MathTex(r"P = \begin{pmatrix} 1 & -0.5 \\ 0.5 & 1 \end{pmatrix}", color="#FFFF00")
        
        grid_content = VGroup(grid_asset, matrix_m).arrange(DOWN)
        self.place_in_area(grid_content, 'E2', 'F5', scale_factor=1.0)
        self.play(FadeIn(grid_asset), Write(matrix_m))
        self.lecture[2].set_color("#FFFF00")

        # === Animation for Lecture Line 4 ===
        # Show column vectors of M as transformed basis vectors
        self.play(Indicate(matrix_m), run_time=1.5)
        self.lecture[3].set_color("#FFFF00")

        # === Animation for Lecture Line 5 ===
        # Pulse Matrix M to emphasize its role as a bridge
        self.play(matrix_m.animate.scale(1.1).set_color("#FFFFFF"), run_time=0.5)
        self.play(matrix_m.animate.scale(1/1.1).set_color("#FFFF00"), run_time=0.5)
        self.lecture[4].set_color("#00FFFF")
