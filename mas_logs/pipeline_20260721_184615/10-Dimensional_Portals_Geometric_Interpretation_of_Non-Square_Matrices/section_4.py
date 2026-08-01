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
        # Initial titles and lines
        title = "Wide Matrices: Stepping Down (3D to 2D)"
        lines = [
            "Wide matrices squash 3D inputs into 2D outputs.",
            "Three input coordinates are projected onto two.",
            "Information is lost as the world flattens.",
            "Think of a 3D bird's shadow on paper.",
            "This is dimensionality reduction in action."
        ]
        
        self.setup_layout(title, lines)

        # Colors from storyboard
        CUBE_COLOR = "#333333"
        MATRIX_COLOR = "#FFFFFF"
        FLAT_COLOR = "#555555"
        VECTOR_COLOR = "#FF00FF"
        SHADOW_COLOR = "#FFFFFF"
        X_COLOR = "#FF0000"
        Y_COLOR = "#00FF00"
        Z_COLOR = "#0000FF"

        # Isometric Projection helper
        def get_iso_point(x, y, z):
            # Simple isometric projection for 2D Scene
            scale = 0.6
            iso_x = (x - y) * 0.866 * scale
            iso_y = ((x + y) * 0.5 + z) * scale
            return np.array([iso_x, iso_y, 0])

        # === Animation for Lecture Line 1 ===
        # Show 3D grid cube #333333 and 2x3 matrix #FFFFFF.
        self.lecture[0].set_color(YELLOW)
        
        cube_edges = VGroup()
        vertices = [[x, y, z] for x in [0, 1] for y in [0, 1] for z in [0, 1]]
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if np.linalg.norm(np.array(vertices[i]) - np.array(vertices[j])) == 1:
                    p1 = get_iso_point(*vertices[i])
                    p2 = get_iso_point(*vertices[j])
                    cube_edges.add(Line(p1, p2, color=CUBE_COLOR))
        
        # Issue 32 fix: Place cube in C2-F4 for better spacing
        self.place_in_area(cube_edges, "C2", "F4", scale_factor=1.0)
        
        # Define origin relative to the placed cube
        cube_origin = cube_edges.get_center() - get_iso_point(0.5, 0.5, 0.5)

        # Issue 31 fix: Matrix in A4-B6 for correct aspect ratio
        matrix = Matrix(
            [["x_1", "y_1", "z_1"], ["x_2", "y_2", "z_2"]], 
            element_to_mobject=MathTex,
            element_to_mobject_config={"font_size": 24}
        ).set_color(MATRIX_COLOR)
        self.place_in_area(matrix, "A4", "B6", scale_factor=0.8)

        self.play(Create(cube_edges), Write(matrix))
        self.wait(2)

        # === Animation for Lecture Line 2 ===
        # Three input coordinates are projected onto two.
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # Color-code columns
        m_entries = matrix.get_entries()
        
        # Axes relative to cube_origin
        axes = VGroup(
            Line(cube_origin + get_iso_point(0,0,0), cube_origin + get_iso_point(1.5,0,0), color=X_COLOR),
            Line(cube_origin + get_iso_point(0,0,0), cube_origin + get_iso_point(0,1.5,0), color=Y_COLOR),
            Line(cube_origin + get_iso_point(0,0,0), cube_origin + get_iso_point(0,0,1.5), color=Z_COLOR)
        )

        self.play(
            m_entries[0].animate.set_color(X_COLOR),
            m_entries[3].animate.set_color(X_COLOR),
            m_entries[1].animate.set_color(Y_COLOR),
            m_entries[4].animate.set_color(Y_COLOR),
            m_entries[2].animate.set_color(Z_COLOR),
            m_entries[5].animate.set_color(Z_COLOR),
            Create(axes)
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Information is lost as the world flattens.
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Create flat version of the cube
        flat_edges = VGroup()
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if np.linalg.norm(np.array(vertices[i]) - np.array(vertices[j])) == 1:
                    v1, v2 = list(vertices[i]), list(vertices[j])
                    v1[2], v2[2] = 0, 0 # Squash Z
                    p1, p2 = get_iso_point(*v1), get_iso_point(*v2)
                    if not np.allclose(p1, p2):
                        flat_edges.add(Line(cube_origin + p1, cube_origin + p2, color=FLAT_COLOR))
        
        self.play(
            Transform(cube_edges, flat_edges),
            axes[2].animate.set_stroke(opacity=0), # Fade out Z axis
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 4 ===
        # Think of a 3D bird's shadow on paper.
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        v_end_raw = [0.7, 0.7, 1.2]
        vector_3d = Arrow(cube_origin + get_iso_point(0,0,0), cube_origin + get_iso_point(*v_end_raw), color=VECTOR_COLOR, buff=0)
        vector_2d = Arrow(cube_origin + get_iso_point(0,0,0), cube_origin + get_iso_point(v_end_raw[0], v_end_raw[1], 0), color=SHADOW_COLOR, buff=0)
        
        label_v = MathTex("\\vec{v}", font_size=24, color=VECTOR_COLOR)
        # Issue 30 fix: Move label to B2 to avoid overlap
        self.place_at_grid(label_v, "B2", scale_factor=0.8)

        self.play(Create(vector_3d), Write(label_v))
        self.wait(1)
        self.play(Create(vector_2d))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # This is dimensionality reduction in action.
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)

        # L031: Use set_fill for MathTex opacity
        self.play(
            FadeOut(vector_3d),
            FadeOut(label_v),
            cube_edges.animate.set_stroke(opacity=0.3),
            vector_2d.animate.set_stroke(width=6),
            m_entries[2].animate.set_fill(opacity=0.3),
            m_entries[5].animate.set_fill(opacity=0.3),
        )
        self.wait(2)
