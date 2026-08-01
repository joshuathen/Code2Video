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

class Section6Scene(TeachingScene):
    def construct(self):
        # Initialize the layout with title and lecture lines
        self.setup_layout("Summary: Thinking with Eyes", [
            "Vectors are movements that define our starting points.",
            "Matrices transform space by moving the basis vectors.",
            "Determinants track how the total area changes."
        ])
        
        # Define colors for elements based on storyboard and lecture context
        HIGHLIGHT_COLOR = "#FFFF00"
        VEC_COLOR = "#00FFFF"
        IHAT_COLOR = "#FF0000"
        JHAT_COLOR = "#00FF00"
        DET_COLOR = "#FFFF00"
        
        # === Animation for Lecture Line 1 ===
        # Lecture: "Vectors are movements that define our starting points."
        self.play(self.lecture[0].animate.set_color(HIGHLIGHT_COLOR))
        
        # [Animation 1] Show a vector arrow rotating smoothly and then scaling longer.
        # Create a vector at the center of the right-side area
        vector = Arrow(ORIGIN, RIGHT * 1.5, color=VEC_COLOR, buff=0)
        self.place_in_area(vector, "B2", "E5", scale_factor=0.8)
        
        self.play(GrowArrow(vector))
        self.play(
            Rotate(vector, angle=PI/3, about_point=vector.get_start()),
            run_time=1.5
        )
        self.play(
            vector.animate.scale(1.4),
            run_time=1.0
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Lecture: "Matrices transform space by moving the basis vectors."
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(HIGHLIGHT_COLOR),
            FadeOut(vector)
        )
        
        # Setup a coordinate plane for transformation demonstration
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=3.5,
            y_length=3.5,
            background_line_style={"stroke_opacity": 0.4}
        )
        # Position plane in the central-left part of the right side grid
        self.place_in_area(plane, "B2", "E4", scale_factor=1.0)
        
        # Basis vectors i-hat and j-hat
        i_hat = Arrow(plane.c2p(0,0), plane.c2p(1,0), color=IHAT_COLOR, buff=0, stroke_width=4)
        j_hat = Arrow(plane.c2p(0,0), plane.c2p(0,1), color=JHAT_COLOR, buff=0, stroke_width=4)
        
        i_label = MathTex(r"\hat{i}", color=IHAT_COLOR).scale(0.7)
        j_label = MathTex(r"\hat{j}", color=JHAT_COLOR).scale(0.7)
        i_label.next_to(i_hat, DOWN, buff=0.1)
        j_label.next_to(j_hat, LEFT, buff=0.1)
        
        # [Animation 2] Fade in i-hat (#FF0000) and j-hat (#00FF00) moving to new positions.
        self.play(Create(plane), FadeIn(i_hat), FadeIn(j_hat), Write(i_label), Write(j_label))
        
        # Matrix to define the transformation: [1 3; 2 1]
        matrix_tex = MathTex(
            r"M = \begin{bmatrix} 1 & 3 \\ 2 & 1 \end{bmatrix}",
            color=WHITE
        ).scale(0.7)
        # Resolved Issue 40: Position at B6 to avoid crowding
        self.place_at_grid(matrix_tex, "B6")
        self.play(Write(matrix_tex))
        self.wait(0.5)
        
        # New positions based on columns of the matrix
        target_i_coords = [1, 2]
        target_j_coords = [3, 1]
        target_i_pos = plane.c2p(*target_i_coords)
        target_j_pos = plane.c2p(*target_j_coords)
        
        self.play(
            i_hat.animate.put_start_and_end_on(plane.c2p(0,0), target_i_pos),
            j_hat.animate.put_start_and_end_on(plane.c2p(0,0), target_j_pos),
            i_label.animate.next_to(target_i_pos, RIGHT, buff=0.1),
            j_label.animate.next_to(target_j_pos, UP, buff=0.1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Lecture: "Determinants track how the total area changes."
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(HIGHLIGHT_COLOR),
        )
        
        # Determinant display
        det_tex = MathTex(r"|\det(M)| = 5", color=DET_COLOR).scale(0.7)
        # Resolved Issue 39: Position at C6 to avoid overlap
        self.place_at_grid(det_tex, "C6")
        
        # [Animation 3] Morph the matrix symbols into the transformed grid lines using a ReplacementTransform.
        
        # Define the matrix transformation for grid points
        def apply_matrix_to_point(coords):
            x, y = coords[0], coords[1]
            new_x = 1*x + 3*y
            new_y = 2*x + 1*y
            return plane.c2p(new_x, new_y)

        # Create a small transformed grid subset
        transformed_grid = VGroup()
        for x in range(-1, 2):
            p1 = apply_matrix_to_point([x, -1])
            p2 = apply_matrix_to_point([x, 1])
            transformed_grid.add(Line(p1, p2, color=BLUE, stroke_opacity=0.4))
        for y in range(-1, 2):
            p1 = apply_matrix_to_point([-1, y])
            p2 = apply_matrix_to_point([1, y])
            transformed_grid.add(Line(p1, p2, color=BLUE, stroke_opacity=0.4))

        # Parallelogram representing the determinant area
        area_polygon = Polygon(
            plane.c2p(0,0), 
            plane.c2p(1,2), 
            plane.c2p(4,3), 
            plane.c2p(3,1),
            fill_opacity=0.4, 
            fill_color=DET_COLOR, 
            stroke_width=0
        )
        
        self.play(
            ReplacementTransform(matrix_tex, transformed_grid),
            Write(det_tex),
            run_time=2
        )
        self.play(FadeIn(area_polygon))
        
        # Highlight the area as the determinant's geometric meaning
        self.play(Indicate(area_polygon, color=DET_COLOR))
        self.wait(3)
