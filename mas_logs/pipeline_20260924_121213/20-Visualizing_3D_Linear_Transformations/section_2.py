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
        self.setup_layout("Defining 3D Transformations", [
            "A linear transformation is defined by basis vector landing.",
            "We represent this using a 3x3 matrix.",
            "Columns show where i, j, and k move."
        ])
        
        # Setup 3D Scene
        # Loading SVG asset as grid
        grid_svg = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2])
        i_vec = Vector(RIGHT, color=RED)
        j_vec = Vector(UP, color=GREEN)
        k_vec = Vector(OUT, color=BLUE)
        basis = VGroup(i_vec, j_vec, k_vec)
        
        three_d_obj = VGroup(grid_svg, axes, basis)
        
        # Layout fix per feedback: use area A4-C6 for 3D object
        self.place_in_area(three_d_obj, 'A4', 'C6', scale_factor=0.6)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        self.play(Create(grid_svg), Create(axes), Create(basis))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Layout fix per feedback: Place matrix at D2, scale 0.6
        matrix = MathTex(r"M = \begin{bmatrix} a & d & g \\ b & e & h \\ c & f & i \end{bmatrix}")
        self.place_at_grid(matrix, 'D2', scale_factor=0.6)
        self.play(Write(matrix))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Animate basis vectors
        new_i = np.array([1.5, 0.5, 0])
        new_j = np.array([-0.5, 1.2, 0])
        new_k = np.array([0, 0.5, 1.5])
        
        self.play(
            i_vec.animate.put_start_and_end_on(ORIGIN, new_i),
            j_vec.animate.put_start_and_end_on(ORIGIN, new_j),
            k_vec.animate.put_start_and_end_on(ORIGIN, new_k),
            run_time=2
        )
        self.wait(2)
