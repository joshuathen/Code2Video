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
            "Build transition matrix P for conversion.",
            "Columns are new basis in standard.",
            "Matrix P maps new coordinates.",
            "Mapping basis coordinates to Cartesian space.",
            "Transition matrix links the two grids."
        ]
        self.setup_layout("The Change of Basis Matrix (P)", lecture_lines)
        
        # Define objects
        matrix_p = MathTex(r"P = \begin{bmatrix} 1 & -1 \\ 1 & 1 \end{bmatrix}", color="#32CD32")
        basis_vectors = VGroup(
            MathTex(r"u = \begin{bmatrix} 1 \\ 1 \end{bmatrix}"),
            MathTex(r"v = \begin{bmatrix} -1 \\ 1 \end{bmatrix}")
        )
        label_p = Text("P", color="#32CD32")
        
        # Load SVG assets
        icon_grid = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        icon_compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#32CD32"))
        self.place_at_grid(icon_grid, "B1", scale_factor=0.5)
        self.play(FadeIn(icon_grid))
        self.place_in_area(matrix_p, 'B3', 'C4', scale_factor=1.2)
        self.play(Write(matrix_p))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#32CD32"))
        self.place_at_grid(basis_vectors[0], 'C3', scale_factor=0.9)
        self.place_at_grid(basis_vectors[1], 'C5', scale_factor=0.9)
        self.play(FadeIn(basis_vectors))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#32CD32"))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#32CD32"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#32CD32"))
        self.place_at_grid(icon_compass, "A5", scale_factor=0.5)
        self.play(FadeIn(icon_compass))
        self.place_at_grid(label_p, "B3", scale_factor=1.0)
        self.play(Write(label_p))
        self.wait(2)
