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

class TeachingScene(ThreeDScene):
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

        # Define fine-grained animation grid (6x6 grid on right side)
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
        lecture_lines = [
            "Tall matrices map low dimensions into high dimensions.",
            "This embedding injects a vector into a larger space.",
            "A 2D shape becomes a 3D thin billboard."
        ]
        self.setup_layout("Mapping Up: Embedding into Higher Dimensions", lecture_lines)
        
        # Colors for lecture lines
        colors = ["#FFADAD", "#CAFFBF", "#FDFFB6"]
        
        # Setup 3D Scene elements
        axes = ThreeDAxes(x_length=4, y_length=4, z_length=4, x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2])
        grid_2d = VGroup(*[Line(start=[-1, i, 0], end=[1, i, 0]) for i in np.arange(-1, 1.1, 0.5)] +
                         [Line(start=[i, -1, 0], end=[i, 1, 0]) for i in np.arange(-1, 1.1, 0.5)])
        grid_2d.set_color(BLUE)
        
        # Billboard asset
        billboard = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billboard.svg")
        
        matrix_tex = MathTex(r"\\begin{pmatrix} a & b \\\\ c & d \\\\ e & f \\end{pmatrix} \\begin{pmatrix} x \\\\ y \\end{pmatrix} = \\begin{pmatrix} X \\\\ Y \\\\ Z \\end{pmatrix}")
        matrix_tex.set_color(WHITE)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(colors[0]))
        self.place_at_grid(axes, 'C3', scale_factor=0.7)
        self.place_at_grid(grid_2d, 'C3', scale_factor=0.45)
        self.place_at_grid(billboard, 'C3', scale_factor=0.3)
        self.play(Create(axes), Create(grid_2d), FadeIn(billboard))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(colors[1]))
        self.place_at_grid(matrix_tex, 'B5', scale_factor=0.6)
        self.play(Write(matrix_tex))

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(colors[2]))
        
        # Transformation: Lift grid into 3D space
        grid_3d = grid_2d.copy().set_color(GREEN)
        self.play(Transform(grid_2d, grid_3d))
        
        # Confirm rotation to show 3D nature
        self.set_camera_orientation(phi=75 * DEGREES, theta=45 * DEGREES)
        self.play(Rotate(billboard, angle=PI/4, axis=OUT))
        self.wait(2)
