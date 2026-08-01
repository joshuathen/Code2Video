from manim import *
import numpy as np

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
        # Data
        title_str = "The DNA of Transformation: Basis Vectors"
        lecture_lines_str = [
            "- We only need to track the two unit vectors.",
            "- i-hat and j-hat serve as the plane's GPS.",
            "- The matrix columns show where these vectors land.",
            "- If i-hat and j-hat move, the world follows.",
            "- Their new coordinates define the entire linear transformation."
        ]
        
        self.setup_layout(title_str, lecture_lines_str)

        # Colors
        RED_I = "#FF0000"
        BLUE_J = "#0000FF"
        GRID_COLOR = "#444444"
        MATRIX_COLOR = "#FFFFFF"
        HIGHLIGHT_COLOR = "#FFFF00"

        # Assets & Objects
        # 1. Plane
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": GRID_COLOR, "stroke_width": 2, "stroke_opacity": 0.6},
            axis_config={"stroke_color": GRID_COLOR, "stroke_width": 2}
        )
        # Resolved Issue 34: Change scale_factor to 0.8
        self.place_in_area(plane, "A1", "F6", scale_factor=0.8)
        plane_origin = plane.coords_to_point(0, 0)

        # 2. Basis Vectors
        # Vectors must be relative to the scene's absolute coordinates calculated via plane.coords_to_point
        i_vec = Vector(plane.coords_to_point(1, 0) - plane_origin, color=RED_I, stroke_width=6).shift(plane_origin)
        j_vec = Vector(plane.coords_to_point(0, 1) - plane_origin, color=BLUE_J, stroke_width=6).shift(plane_origin)
        
        i_label = MathTex(r"\hat{i}", color=RED_I, font_size=28)
        j_label = MathTex(r"\hat{j}", color=BLUE_J, font_size=28)
        
        # Initial positions for labels
        i_label.next_to(i_vec.get_end(), DOWN, buff=0.1)
        j_label.next_to(j_vec.get_end(), LEFT, buff=0.1)

        # 3. Matrix
        matrix_vgroup = Matrix([[1, -1], [1, 1]], 
                             left_bracket="[", 
                             right_bracket="]",
                             element_to_mobject_config={"color": WHITE})
        m_label = MathTex("M = ", color=WHITE, font_size=36)
        matrix_full = VGroup(m_label, matrix_vgroup).arrange(RIGHT, buff=0.2)
        
        # Resolved Issue 33: Move to B2 and scale to 0.7
        self.place_at_grid(matrix_full, "B2", scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        # "- We only need to track the two unit vectors."
        self.lecture[0].set_color(HIGHLIGHT_COLOR)
        self.add(plane)
        self.play(
            GrowArrow(i_vec),
            GrowArrow(j_vec),
            Write(i_label),
            Write(j_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "- i-hat and j-hat serve as the plane's GPS."
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(HIGHLIGHT_COLOR)
        
        # Pulse animation
        self.play(
            i_vec.animate.scale(1.2, about_point=plane_origin).set_stroke(width=10),
            j_vec.animate.scale(1.2, about_point=plane_origin).set_stroke(width=10),
            rate_func=there_and_back,
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "- The matrix columns show where these vectors land."
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(HIGHLIGHT_COLOR)
        
        self.play(Write(matrix_full))
        
        c1 = matrix_vgroup.get_columns()[0]
        c2 = matrix_vgroup.get_columns()[1]
        
        r1 = SurroundingRectangle(c1, color=RED_I, buff=0.1)
        r2 = SurroundingRectangle(c2, color=BLUE_J, buff=0.1)
        
        self.play(Create(r1))
        self.wait(0.5)
        self.play(ReplacementTransform(r1, r2))
        self.wait(0.5)
        self.play(FadeOut(r2))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "- If i-hat and j-hat move, the world follows."
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(HIGHLIGHT_COLOR)
        
        # Target physical coordinates calculated BEFORE transformation
        target_i_pos = plane.coords_to_point(1, 1)
        target_j_pos = plane.coords_to_point(-1, 1)
        matrix_vals = [[1, -1], [1, 1]]
        
        # Update labels to follow the vectors
        i_label.add_updater(lambda m: m.next_to(i_vec.get_end(), RIGHT, buff=0.1))
        j_label.add_updater(lambda m: m.next_to(j_vec.get_end(), LEFT, buff=0.1))

        self.play(
            plane.animate.apply_matrix(matrix_vals, about_point=plane_origin),
            i_vec.animate.put_start_and_end_on(plane_origin, target_i_pos),
            j_vec.animate.put_start_and_end_on(plane_origin, target_j_pos),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "- Their new coordinates define the entire linear transformation."
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(HIGHLIGHT_COLOR)
        
        i_flash = Flash(i_vec.get_end(), color=RED_I, flash_radius=0.5, line_length=0.2)
        j_flash = Flash(j_vec.get_end(), color=BLUE_J, flash_radius=0.5, line_length=0.2)
        
        self.play(i_flash, j_flash)
        
        # Clean up updaters
        i_label.clear_updaters()
        j_label.clear_updaters()
        
        self.wait(2)
