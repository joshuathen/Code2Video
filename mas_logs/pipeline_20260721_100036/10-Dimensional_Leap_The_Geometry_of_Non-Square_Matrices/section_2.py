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
        # Data from storyboard
        title_text = "Prerequisite Check: The DNA of a Matrix"
        lecture_lines = [
            "Columns represent the number of input basis vectors.",
            "Rows represent coordinates needed in the output space.",
            "This mapping defines the dimension of the transformation."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        COLOR_I = "#FF0000"
        COLOR_J = "#00FF00"
        COLOR_MATRIX = "#FFA500"
        COLOR_VEC = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # Columns represent the number of input basis vectors.
        self.play(self.lecture[0].animate.set_color(COLOR_I))
        
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=3.5,
            y_length=3.5,
            background_line_style={"stroke_opacity": 0.2}
        )
        # Fix Issue 20: Position plane better (B2-E6)
        self.place_in_area(plane, 'B2', 'E6', scale_factor=0.8)
        
        origin = plane.get_origin()
        # Create vectors relative to the plane's coordinate system
        i_hat = Vector(plane.c2p(1, 0) - origin, color=COLOR_I).shift(origin)
        j_hat = Vector(plane.c2p(0, 1) - origin, color=COLOR_J).shift(origin)
        
        i_label = MathTex(r"\hat{i}", color=COLOR_I, font_size=24).next_to(i_hat, DOWN, buff=0.1)
        j_label = MathTex(r"\hat{j}", color=COLOR_J, font_size=24).next_to(j_hat, LEFT, buff=0.1)
        
        self.play(Create(plane))
        self.play(GrowArrow(i_hat), Write(i_label))
        self.play(GrowArrow(j_hat), Write(j_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rows represent coordinates needed in the output space.
        self.play(self.lecture[1].animate.set_color(COLOR_MATRIX))
        
        # Transition to matrix representation
        # Display a 3x2 matrix
        matrix_mobject = Matrix(
            [["v_{1x}", "v_{2x}"], ["v_{1y}", "v_{2y}"], ["v_{1z}", "v_{2z}"]],
            left_bracket="[", right_bracket="]"
        ).set_color(WHITE)
        # Fix Issue 21: Avoid crowding at the top (B2-D5)
        self.place_in_area(matrix_mobject, 'B2', 'D5', scale_factor=0.8)
        
        self.play(
            FadeOut(plane, i_hat, j_hat, i_label, j_label),
            Write(matrix_mobject)
        )
        
        col1 = matrix_mobject.get_columns()[0]
        col2 = matrix_mobject.get_columns()[1]
        
        rect1 = SurroundingRectangle(col1, color=COLOR_I, buff=0.1)
        rect2 = SurroundingRectangle(col2, color=COLOR_J, buff=0.1)
        
        label_v1 = MathTex(r"\vec{v}_1", color=COLOR_I, font_size=24).next_to(rect1, DOWN, buff=0.1)
        label_v2 = MathTex(r"\vec{v}_2", color=COLOR_J, font_size=24).next_to(rect2, DOWN, buff=0.1)
        
        self.play(Create(rect1), Write(label_v1))
        self.play(Create(rect2), Write(label_v2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This mapping defines the dimension of the transformation.
        self.play(self.lecture[2].animate.set_color(COLOR_VEC))
        
        # Show vector [x, y] as a linear combination of columns.
        lin_comb = MathTex(
            "x", r"\vec{v}_1", "+", "y", r"\vec{v}_2", "=", r"\vec{w}",
            color=COLOR_VEC, font_size=32
        )
        # Fix Issue 19: Prevent overlap with matrix labels (F2-F5)
        self.place_in_area(lin_comb, 'F2', 'F5', scale_factor=0.8)
        
        self.play(Write(lin_comb))
        
        # Connection highlight between columns and linear combination terms
        # Indices: 0: x, 1: \vec{v}_1, 2: +, 3: y, 4: \vec{v}_2, 5: =, 6: \vec{w}
        self.play(
            Indicate(col1, color=COLOR_I),
            Indicate(lin_comb[1], color=COLOR_I),
            run_time=2
        )
        self.play(
            Indicate(col2, color=COLOR_J),
            Indicate(lin_comb[4], color=COLOR_J),
            run_time=2
        )
        
        self.wait(2)
