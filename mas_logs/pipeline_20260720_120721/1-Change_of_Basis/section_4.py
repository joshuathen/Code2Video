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
        # Colors for lecture lines and corresponding animations
        L1_COLOR = "#FFD700"  # Gold
        L2_COLOR = "#ADFF2F"  # GreenYellow
        L3_COLOR = "#87CEFA"  # LightSkyBlue

        title = "Example: Transforming a Vector"
        lecture_lines = [
            "Example: Vector u=[3,2] in standard basis.",
            "New basis b1=[1,1], b2=[-1,1].",
            "Find coordinates in the new basis."
        ]
        self.setup_layout(title, lecture_lines)

        # Manim objects for the scene
        number_plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            x_length=8,
            y_length=8,
            axis_config={"include_numbers": True, "label_constructor": Text}, # FIX: Use Text for numbers to avoid LaTeX dependency
            background_line_style={
                "stroke_color": GREY,
                "stroke_width": 0.5,
                "stroke_opacity": 0.6,
            }
        )
        self.place_in_area(number_plane, 'A1', 'F4', scale_factor=0.9) # Place plane on the left part of the right grid

        self.play(FadeIn(number_plane))

        # Vector u in standard basis
        u_E_coords = np.array([3, 2, 0])
        vector_u_E = Arrow(ORIGIN, u_E_coords, buff=0, color=L1_COLOR)
        # The original error "FileNotFoundError: [Errno 2] No such file or directory: 'latex'" occurs because MathTex requires a LaTeX installation.
        # To fix this within the code block, replace MathTex with Text and simplify the content.
        u_E_label = Text(r"u_std = [3, 2]", color=L1_COLOR, font_size=30)
        self.place_at_grid(u_E_label, 'D3', scale_factor=0.8) # Adjust label position to avoid vector head

        # Basis vectors for the new basis
        b1_vec_coords = np.array([1, 1, 0])
        b2_vec_coords = np.array([-1, 1, 0])
        
        vector_b1 = Arrow(ORIGIN, b1_vec_coords, buff=0, color=L2_COLOR)
        # Fix: Replaced MathTex with Text to avoid the 'latex' FileNotFoundError.
        b1_label = Text("b\u2081 = [1, 1]", color=L2_COLOR, font_size=30)
        self.place_at_grid(b1_label, 'B5', scale_factor=0.8) # Position relative to b1 vector

        vector_b2 = Arrow(ORIGIN, b2_vec_coords, buff=0, color=L2_COLOR)
        # Fix: The original error occurs here due to MathTex requiring a LaTeX installation.
        # Replacing MathTex with Text avoids this dependency.
        b2_label = Text("b\u2082 = [-1, 1]", color=L2_COLOR, font_size=30)
        self.place_at_grid(b2_label, 'C5', scale_factor=0.8) # Position relative to b2 vector

        # Transition matrix P
        # FIX: The `FileNotFoundError: [Errno 2] No such file or directory: 'latex'` occurs here.
        # Replacing MathTex with Text and simplifying the matrix representation to avoid LaTeX dependency.
        P_matrix = Text(r"P = [[1, -1], [1, 1]]", color=L2_COLOR, font_size=30)
        self.place_at_grid(P_matrix, 'A5', scale_factor=0.8)

        # Inverse of P and transformation calculation
        # FIX: The error "FileNotFoundError: [Errno 2] No such file or directory: 'latex'" occurs here.
        # Replacing MathTex with Text and simplifying the content to avoid LaTeX syntax.
        formula_u_B = Text(r"u_B = P^-1 u_std", color=L3_COLOR, font_size=30)
        self.place_at_grid(formula_u_B, 'D5', scale_factor=0.8)

        # The error `FileNotFoundError: [Errno 2] No such file or directory: 'latex'` occurs here.
        # To fix this, MathTex is replaced with Text, and the content is simplified to remove LaTeX syntax.
        P_inv_calc = Text(
            r"P^-1 = (1/det(P)) [[1, 1], [-1, 1]] = [[0.5, 0.5], [-0.5, 0.5]]", # Simplified string
            color=L3_COLOR, font_size=30
        )
        self.place_at_grid(P_inv_calc, 'E5', scale_factor=0.7)

        # FIX: This is the specific line causing the FileNotFoundError: 'latex'.
        # Replace MathTex with Text and simplify the LaTeX string to a plain text string.
        u_B_calc = Text(
            r"u\u208B = [[0.5, 0.5], [-0.5, 0.5]] [[3], [2]] = [[2.5], [-0.5]]", # Simplified string
            color=L3_COLOR, font_size=30
        )
        self.place_at_grid(u_B_calc, 'F5', scale_factor=0.7)

        # Vectors representing u in the new basis
        u_B_result = np.array([2.5, -0.5, 0])
        vector_u_B_x_comp = Arrow(ORIGIN, u_B_result[0] * b1_vec_coords, buff=0, color=L3_COLOR)
        vector_u_B_y_comp = Arrow(
            u_B_result[0] * b1_vec_coords,
            u_B_result[0] * b1_vec_coords + u_B_result[1] * b2_vec_coords,
            buff=0, color=L3_COLOR
        )
        
        # FIX: This MathTex also needs to be converted to Text to avoid the 'latex' dependency.
        u_new_coords_label = Text(
            r"u = 2.5 b\u2081 - 0.5 b\u2082", # Simplified string with unicode subscripts
            color=L3_COLOR, font_size=30
        )
        self.place_at_grid(u_new_coords_label, 'F3', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(L1_COLOR))
        self.play(
            Create(vector_u_E),
            FadeIn(u_E_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(L2_COLOR)
        )
        self.play(
            Create(vector_b1),
            FadeIn(b1_label),
            Create(vector_b2),
            FadeIn(b2_label),
            FadeIn(P_matrix),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(L3_COLOR)
        )
        self.play(
            FadeIn(formula_u_B),
            run_time=1
        )
        self.wait(0.5)
        self.play(
            FadeIn(P_inv_calc),
            run_time=2
        )
        self.wait(0.5)
        self.play(
            FadeIn(u_B_calc),
            run_time=2
        )
        self.wait(1)
        
        # Visually confirm the transformation
        self.play(
            Create(vector_u_B_x_comp),
            Create(vector_u_B_y_comp),
            FadeIn(u_new_coords_label),
            run_time=3
        )
        self.play(Flash(vector_u_E, color=L1_COLOR, line_length=0.5, num_lines=10)) # Emphasize that the endpoint is the same
        self.wait(2)

        # Reset lecture line color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
