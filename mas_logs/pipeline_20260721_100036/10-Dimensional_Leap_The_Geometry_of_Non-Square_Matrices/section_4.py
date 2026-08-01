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

class Section4Scene(TeachingScene):
    def construct(self):
        # Setup Title and Lecture Lines
        title_text = "Wide Matrices: The Great Squashing (3D to 2D)"
        lecture_lines = [
            "A 2x3 matrix maps 3D input to 2D output.",
            "Three basis vectors are squashed into a 2D plane.",
            "This process inevitably involves losing one dimension of information.",
            "Think of it as projecting a 3D bird's shadow.",
            "The Null Space contains vectors that collapse to zero."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors from storyboard and script alignment
        COLOR_MATRIX = "#FFFF00"
        COLOR_PLANE = "#808080"
        COLOR_SQUASH = "#FF0000"
        COLOR_CUBE = "#FFFFFF"
        COLOR_NULL = "#00FFFF"

        # === Animation for Lecture Line 1 ===
        # Line 1: A 2x3 matrix maps 3D input to 2D output. (COLOR_MATRIX)
        self.lecture[0].set_color(COLOR_MATRIX)
        matrix_tex = MathTex(
            "A_{2 \\times 3} = \\begin{bmatrix} 1 & 0 & 0 \\\\ 0 & 1 & 0 \\end{bmatrix}",
            color=COLOR_MATRIX
        )
        # Resolved Issue 25: Move matrix_tex to A4
        self.place_at_grid(matrix_tex, "A4", scale_factor=0.9)

        # Helper for a pseudo-3D wireframe cube
        def create_cube(side=1.0, color=WHITE):
            f_tl = np.array([-side/2, side/2, 0])
            f_tr = np.array([side/2, side/2, 0])
            f_bl = np.array([-side/2, -side/2, 0])
            f_br = np.array([side/2, -side/2, 0])
            
            off = side * 0.4
            b_tl = f_tl + np.array([off, off, 0])
            b_tr = f_tr + np.array([off, off, 0])
            b_bl = f_bl + np.array([off, off, 0])
            b_br = f_br + np.array([off, off, 0])
            
            edges = VGroup(
                Line(f_tl, f_tr), Line(f_tr, f_br), Line(f_br, f_bl), Line(f_bl, f_tl), # Front
                Line(b_tl, b_tr), Line(b_tr, b_br), Line(b_br, b_bl), Line(b_bl, b_tl), # Back
                Line(f_tl, b_tl), Line(f_tr, b_tr), Line(f_br, b_br), Line(f_bl, b_bl)  # Connectors
            ).set_color(color)
            return edges

        cube = create_cube(side=1.0, color=COLOR_CUBE)
        # Resolved Issue 27: Expand area to B3-E5
        self.place_in_area(cube, "B3", "E5", scale_factor=0.8)
        
        self.play(Write(matrix_tex), Create(cube))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2: Three basis vectors are squashed into a 2D plane. (COLOR_PLANE)
        self.lecture[1].set_color(COLOR_PLANE)
        
        # Plane represented as a parallelogram below the cube
        shadow_plane = Polygon(
            self.grid["E1"], self.grid["E5"], self.grid["F6"], self.grid["F2"],
            color=COLOR_PLANE, fill_opacity=0.2, stroke_width=2
        )
        # Re-calculating center based on grid positions to ensure it stays in area
        shadow_plane_center = (self.grid["E1"] + self.grid["E5"] + self.grid["F6"] + self.grid["F2"]) / 4
        shadow_plane.move_to(shadow_plane_center)
        
        # Represent basis vectors (X, Y, Z) starting from cube center
        origin = cube.get_center()
        vec_x = Arrow(origin, origin + RIGHT*0.6, color=RED, buff=0, stroke_width=3)
        vec_y = Arrow(origin, origin + UP*0.6, color=GREEN, buff=0, stroke_width=3)
        vec_z = Arrow(origin, origin + np.array([0.3, 0.3, 0]), color=BLUE, buff=0, stroke_width=3)
        basis_group = VGroup(vec_x, vec_y, vec_z)

        self.play(Create(shadow_plane), Create(basis_group))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Line 3: This process inevitably involves losing one dimension of information. (COLOR_SQUASH)
        self.lecture[2].set_color(COLOR_SQUASH)
        
        # Flattening: The cube and basis vectors collapse into a flat square on the plane
        shadow_square = Rectangle(width=1.0, height=0.8, color=COLOR_SQUASH, fill_opacity=0.4)
        shadow_square.move_to(shadow_plane.get_center())
        
        self.play(
            cube.animate.scale(0.1).move_to(shadow_plane.get_center()).set_color(COLOR_SQUASH),
            basis_group.animate.scale(0.1).move_to(shadow_plane.get_center()).set_color(COLOR_SQUASH),
            FadeOut(cube),
            FadeOut(basis_group),
            FadeIn(shadow_square),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Line 4: Think of it as projecting a 3D bird's shadow. (COLOR_CUBE)
        self.lecture[3].set_color(COLOR_CUBE)
        
        # Re-show the "bird" cube high up to illustrate projection rays
        bird_cube = create_cube(side=0.8, color=COLOR_CUBE)
        self.place_at_grid(bird_cube, "B3")
        
        # Representative projection rays from corners
        rays = VGroup(
            Line(bird_cube.get_center() + np.array([-0.4, 0.4, 0]), shadow_square.get_corner(UL), color=WHITE, stroke_opacity=0.3),
            Line(bird_cube.get_center() + np.array([0.4, 0.4, 0]), shadow_square.get_corner(UR), color=WHITE, stroke_opacity=0.3),
            Line(bird_cube.get_center() + np.array([-0.4, -0.4, 0]), shadow_square.get_corner(DL), color=WHITE, stroke_opacity=0.3),
            Line(bird_cube.get_center() + np.array([0.4, -0.4, 0]), shadow_square.get_corner(DR), color=WHITE, stroke_opacity=0.3)
        )
        
        self.play(FadeIn(bird_cube), Create(rays))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Line 5: The Null Space contains vectors that collapse to zero. (COLOR_NULL)
        self.lecture[4].set_color(COLOR_NULL)
        
        # The Null Space vector (the Z axis that maps to zero)
        null_vec = Arrow(bird_cube.get_center(), bird_cube.get_center() + np.array([0.3, 0.3, 0]), color=COLOR_NULL, buff=0)
        null_label = Text("Collapses to point", font_size=18, color=COLOR_NULL)
        # Resolved Issue 26: Move null_label to E6
        self.place_at_grid(null_label, "E6", scale_factor=0.8)
        
        # The target point in the 2D output
        zero_dot = Dot(shadow_square.get_center(), color=COLOR_NULL)
        
        self.play(Create(null_vec), Write(null_label))
        self.play(
            null_vec.animate.scale(0.01).move_to(shadow_square.get_center()),
            FadeIn(zero_dot),
            run_time=2
        )
        self.wait(2)
