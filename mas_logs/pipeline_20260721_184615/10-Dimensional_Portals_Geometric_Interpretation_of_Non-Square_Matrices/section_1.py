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

class Section1Scene(TeachingScene):
    def construct(self):
        # Fetching data from shared state
        # Title: Prerequisite Refresher: The Basis of Motion
        # Line 1: Square matrices transform input vectors within the same space.
        # Line 2: Basis vectors track where every point lands.
        # Line 3: These unit vectors are the DNA of transformations.
        
        self.setup_layout(
            "Prerequisite Refresher: The Basis of Motion",
            [
                "Square matrices transform input vectors within the same space.",
                "Basis vectors track where every point lands.",
                "These unit vectors are the DNA of transformations."
            ]
        )
        
        # === Animation for Lecture Line 1 ===
        # Show 2D grid #333333 with 2x2 matrix text A = [[a,b],[c,d]] #FFFFFF. Matrix appears with FadeIn.
        self.lecture[0].set_color("#FFFFFF")
        
        grid = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={
                "stroke_color": "#333333",
                "stroke_width": 2,
                "stroke_opacity": 0.8
            },
            axis_config={"include_tip": False}
        )
        # Position grid in area B3 to F6 to avoid lecture overlap (Fix for Issue 22)
        self.place_in_area(grid, "B3", "F6", scale_factor=0.7)
        
        matrix_tex = MathTex(
            r"A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}",
            color="#FFFFFF"
        )
        # Place matrix at A5 (Fix for Issue 23)
        self.place_at_grid(matrix_tex, "A5", scale_factor=0.8)
        
        self.play(Create(grid), run_time=1.5)
        self.play(FadeIn(matrix_tex))
        self.wait(2.0)
        
        # === Animation for Lecture Line 2 ===
        # Create i-hat [1,0] vector #FF0000 and j-hat [0,1] vector #00FF00. Add labels near heads.
        self.lecture[0].set_color("#666666") # Dim previous line
        self.lecture[1].set_color("#FFFF00") # Highlight Line 2
        
        # Basis vectors relative to grid
        i_hat = Arrow(grid.c2p(0, 0), grid.c2p(1, 0), buff=0, color="#FF0000")
        j_hat = Arrow(grid.c2p(0, 0), grid.c2p(0, 1), buff=0, color="#00FF00")
        
        i_label = MathTex(r"\hat{i}", color="#FF0000").scale(0.7)
        j_label = MathTex(r"\hat{j}", color="#00FF00").scale(0.7)
        
        # Position labels near heads
        i_label.next_to(i_hat.get_end(), RIGHT, buff=0.1)
        j_label.next_to(j_hat.get_end(), UP, buff=0.1)
        
        self.play(GrowArrow(i_hat), Write(i_label))
        self.play(GrowArrow(j_hat), Write(j_label))
        self.wait(2.0)
        
        # === Animation for Lecture Line 3 ===
        # Apply matrix transformation to grid and vectors. Grid warps, vectors move to [a,c] and [b,d].
        self.lecture[1].set_color("#666666")
        self.lecture[2].set_color("#00FFFF") # Highlight Line 3
        
        # Concrete matrix for visual clarity
        # A = [[2, 1], [0.5, 1.5]]
        matrix_concrete = MathTex(
            r"A = \begin{bmatrix} 2 & 1 \\ 0.5 & 1.5 \end{bmatrix}",
            color="#FFFFFF"
        )
        # Place concrete matrix at A5 (Fix for Issue 24)
        self.place_at_grid(matrix_concrete, "A5", scale_factor=0.8)
        
        self.play(Transform(matrix_tex, matrix_concrete))
        self.wait(1.5)
        
        # Transformation matrix values
        m_vals = [[2, 1], [0.5, 1.5]]
        
        # Function to apply the transformation relative to the grid origin
        grid_origin = grid.c2p(0, 0)
        def transform_point(point):
            p = point - grid_origin
            new_x = m_vals[0][0] * p[0] + m_vals[0][1] * p[1]
            new_y = m_vals[1][0] * p[0] + m_vals[1][1] * p[1]
            return np.array([new_x, new_y, 0]) + grid_origin

        # Updaters for labels to follow the vectors during transformation
        i_label.add_updater(lambda m: m.next_to(i_hat.get_end(), RIGHT, buff=0.1))
        j_label.add_updater(lambda m: m.next_to(j_hat.get_end(), UP, buff=0.1))
        
        self.play(
            grid.animate.apply_function(transform_point),
            i_hat.animate.apply_function(transform_point),
            j_hat.animate.apply_function(transform_point),
            run_time=3,
            rate_func=rate_functions.ease_in_out_sine # Using rate_functions prefix (L024)
        )
        self.wait(2.5)
        
        # Final cleanup: remove updaters
        i_label.clear_updaters()
        j_label.clear_updaters()
