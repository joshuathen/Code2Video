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

class Section2Scene(TeachingScene):
    def construct(self):
        title = "Prerequisite Knowledge: Basis Vectors"
        lines = [
            'We track space using basis vectors i-hat and j-hat.',
            'A matrix records where these basis vectors land.',
            'They define the shape of the newly transformed grid.'
        ]
        self.setup_layout(title, lines)

        # Helper to create a bracketed matrix without Tex
        def create_matrix(a, c, b, d, color=WHITE):
            content = VGroup(
                VGroup(Text(str(a), font_size=24), Text(str(c), font_size=24)).arrange(RIGHT, buff=0.5),
                VGroup(Text(str(b), font_size=24), Text(str(d), font_size=24)).arrange(RIGHT, buff=0.5)
            ).arrange(DOWN, buff=0.3)
            
            bracket_l = Text("[", font_size=60).scale(1.2).next_to(content, LEFT, buff=0.1)
            bracket_r = Text("]", font_size=60).scale(1.2).next_to(content, RIGHT, buff=0.1)
            matrix = VGroup(content, bracket_l, bracket_r).set_color(color)
            return matrix

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Create a coordinate system
        # Center the plane at B2-E5 area (Issue 37)
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.4},
            axis_config={"include_tip": True}
        )
        self.place_in_area(plane, 'B2', 'E5', scale_factor=0.9)
        
        # i-hat (1,0) - Red
        i_hat = Arrow(
            start=plane.coords_to_point(0, 0),
            end=plane.coords_to_point(1, 0),
            buff=0,
            color="#FF0000"
        )
        i_label = Text("i", font_size=18, color="#FF0000").next_to(i_hat, DOWN, buff=0.1)
        
        # j-hat (0,1) - Green
        j_hat = Arrow(
            start=plane.coords_to_point(0, 0),
            end=plane.coords_to_point(0, 1),
            buff=0,
            color="#00FF00"
        )
        j_label = Text("j", font_size=18, color="#00FF00").next_to(j_hat, LEFT, buff=0.1)
        
        self.play(Create(plane), GrowArrow(i_hat), GrowArrow(j_hat), FadeIn(i_label), FadeIn(j_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Matrix values
        a, b = 2, 1
        c, d = 0, 2
        
        matrix_obj = create_matrix("a", "c", "b", "d", color=WHITE)
        # Issue 35: place in area A3-A4, scale 0.6
        self.place_in_area(matrix_obj, 'A3', 'A4', scale_factor=0.6)
        
        # Highlight columns for i-hat and j-hat
        col1_rect = SurroundingRectangle(matrix_obj[0][0], color="#FF0000", buff=0.1)
        col2_rect = SurroundingRectangle(matrix_obj[0][1], color="#00FF00", buff=0.1)
        
        self.play(FadeIn(matrix_obj))
        self.play(Create(col1_rect))
        self.wait(0.5)
        self.play(ReplacementTransform(col1_rect, col2_rect))
        self.wait(0.5)
        self.play(FadeOut(col2_rect))
        
        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Destination points
        target_i = plane.coords_to_point(a, b)
        target_j = plane.coords_to_point(c, d)
        
        # Generic to concrete matrix labels
        concrete_matrix = create_matrix(str(a), str(c), str(b), str(d), color=WHITE)
        # Issue 36: place in area A3-A4, scale 0.6
        self.place_in_area(concrete_matrix, 'A3', 'A4', scale_factor=0.6)
        
        # Transform labels and vectors
        self.play(ReplacementTransform(matrix_obj, concrete_matrix))
        
        self.play(
            i_hat.animate.put_start_and_end_on(plane.coords_to_point(0, 0), target_i),
            j_hat.animate.put_start_and_end_on(plane.coords_to_point(0, 0), target_j),
            i_label.animate.next_to(target_i, RIGHT, buff=0.1),
            j_label.animate.next_to(target_j, UP, buff=0.1),
            run_time=2
        )
        
        # Visualizing the "shape" - simple parallelogram
        parallelogram = Polygon(
            plane.coords_to_point(0, 0),
            target_i,
            plane.coords_to_point(a + c, b + d),
            target_j,
            stroke_width=2,
            fill_opacity=0.2,
            color=BLUE
        )
        
        self.play(Create(parallelogram))
        self.wait(2)
