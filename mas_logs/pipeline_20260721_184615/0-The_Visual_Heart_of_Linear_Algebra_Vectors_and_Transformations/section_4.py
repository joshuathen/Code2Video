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

class Section4Scene(TeachingScene):
    def construct(self):
        # === Setup ===
        title_text = "Matrices as a Map of Basis Vectors"
        lecture_lines = [
            "A matrix tracks where basis vectors land after transformation.",
            "The first column shows i-hat's new coordinates.",
            "The second column tracks j-hat's new landing spot.",
            "Knowing these two reveals where every point moves.",
            "The matrix is a GPS for the entire plane."
        ]
        self.setup_layout(title_text, lecture_lines)

        # Matrix setup for [[0, -1], [1, 0]]
        matrix_val = [[0, -1], [1, 0]]
        matrix_mob = Matrix(matrix_val, 
                           element_to_mobject_config={"color": "#00FFFF"},
                           bracket_config={"color": "#00FFFF"}).set_color("#00FFFF")
        self.place_in_area(matrix_mob, 'A3', 'A4', scale_factor=0.8)

        # Coordinate Plane Setup
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            axis_config={"stroke_color": "#FFFFFF", "stroke_width": 2},
            background_line_style={"stroke_color": "#444444", "stroke_width": 1, "stroke_opacity": 0.5}
        )
        
        # Vectors and Labels
        i_hat = Arrow(ORIGIN, RIGHT, buff=0, color="#FF0000", stroke_width=4)
        j_hat = Arrow(ORIGIN, UP, buff=0, color="#00FF00", stroke_width=4)
        
        i_label = MathTex(r"\hat{i}", color="#FF0000")
        j_label = MathTex(r"\hat{j}", color="#00FF00")

        # Create plane group for initial positioning
        plane_group = VGroup(plane, i_hat, j_hat, i_label, j_label)
        self.place_in_area(plane_group, 'C2', 'F5', scale_factor=0.9)
        
        # Calculate origin position for future vector placement
        # We use .copy() to keep the reference static even if the mobject changes
        origin_pos = plane.get_center().copy()
        
        # Refine label positions using the grid
        self.place_at_grid(i_label, 'D5', scale_factor=0.7)
        self.place_at_grid(j_label, 'C4', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        # "A matrix tracks where basis vectors land after transformation."
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        self.wait(1.5)
        self.play(
            FadeIn(matrix_mob),
            FadeIn(plane),
            GrowArrow(i_hat),
            GrowArrow(j_hat),
            Write(i_label),
            Write(j_label),
            run_time=1.5
        )
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        # "The first column shows i-hat's new coordinates."
        self.play(
            self.lecture[0].animate.set_color("#FFFFFF"),
            self.lecture[1].animate.set_color("#FF0000")
        )
        self.wait(1.5)
        
        # Highlight first column
        col1_rect = SurroundingRectangle(matrix_mob.get_columns()[0], color="#FF0000", buff=0.1)
        self.play(Create(col1_rect))
        
        # i-hat moves from (1,0) to (0,1)
        i_hat_target = Arrow(origin_pos, origin_pos + UP, buff=0, color="#FF0000", stroke_width=4)
        i_label_target = MathTex(r"\hat{i}'", color="#FF0000")
        self.place_at_grid(i_label_target, 'C4', scale_factor=0.7)
        
        self.play(
            Transform(i_hat, i_hat_target),
            Transform(i_label, i_label_target),
            run_time=2
        )
        self.wait(2.0)

        # === Animation for Lecture Line 3 ===
        # "The second column tracks j-hat's new landing spot."
        self.play(
            self.lecture[1].animate.set_color("#FFFFFF"),
            self.lecture[2].animate.set_color("#00FF00")
        )
        self.wait(1.5)

        # Highlight second column
        col2_rect = SurroundingRectangle(matrix_mob.get_columns()[1], color="#00FF00", buff=0.1)
        self.play(ReplacementTransform(col1_rect, col2_rect))

        # j-hat moves from (0,1) to (-1,0)
        j_hat_target = Arrow(origin_pos, origin_pos + LEFT, buff=0, color="#00FF00", stroke_width=4)
        j_label_target = MathTex(r"\hat{j}'", color="#00FF00")
        self.place_at_grid(j_label_target, 'D3', scale_factor=0.7)

        self.play(
            Transform(j_hat, j_hat_target),
            Transform(j_label, j_label_target),
            run_time=2
        )
        self.wait(2.0)

        # === Animation for Lecture Line 4 ===
        # "Knowing these two reveals where every point moves."
        self.play(
            self.lecture[2].animate.set_color("#FFFFFF"),
            self.lecture[3].animate.set_color("#FFFF00")
        )
        self.wait(2.0)
        
        self.play(FadeOut(col2_rect))
        
        # Morph grid (90-degree CCW rotation)
        self.play(
            plane.animate.apply_matrix([[0, -1], [1, 0]], about_point=origin_pos),
            run_time=2
        )
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        # "The matrix is a GPS for the entire plane."
        self.play(
            self.lecture[3].animate.set_color("#FFFFFF"),
            self.lecture[4].animate.set_color("#00FFFF")
        )
        self.wait(1.5)
        
        col1_rect_final = SurroundingRectangle(matrix_mob.get_columns()[0], color="#FF0000", buff=0.1)
        col2_rect_final = SurroundingRectangle(matrix_mob.get_columns()[1], color="#00FF00", buff=0.1)
        
        self.play(Create(col1_rect_final), Create(col2_rect_final))
        self.play(Indicate(i_hat), Indicate(j_hat))
        self.wait(3.0)
