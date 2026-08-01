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
        # Initial Setup
        title = "The Characteristic Equation: Finding the Secret Directions"
        lines = [
            'We solve Av equals \u03bbv for these vectors.', 
            'Rearranging gives the determinant of A minus \u03bbI.', 
            'Setting the determinant to zero collapses the space.', 
            'This collapse identifies our valid eigenvalues.', 
            'Then, finding the null space yields eigenvectors.'
        ]
        self.setup_layout(title, lines)

        # Colors
        COLOR_EQ = "#FFFFFF"
        COLOR_PLANE = "#1890FF"
        COLOR_HIGHLIGHT = "#F5222D"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Av = lambda v
        eq1 = Text("A v = \u03bb v", font_size=36, color=COLOR_EQ)
        self.place_at_grid(eq1, "B2", scale_factor=1.0)
        self.play(FadeIn(eq1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)

        # (A - lambda I)v = 0
        eq2 = Text("(A - \u03bb I) v = 0", font_size=36, color=COLOR_EQ)
        self.place_at_grid(eq2, "B2", scale_factor=1.0)
        
        self.play(ReplacementTransform(eq1, eq2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)

        # Initial NumberPlane
        plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_color": COLOR_PLANE, "stroke_opacity": 0.5},
            axis_config={"stroke_color": COLOR_PLANE, "stroke_width": 2}
        )
        self.place_in_area(plane, "D2", "F6", scale_factor=0.6)
        
        mat_label = Text("A - \u03bb I =", font_size=24, color=WHITE)
        self.place_at_grid(mat_label, "A5", scale_factor=0.8)
        
        def get_mat_group(lam_val):
            # Matrix: [[3-lambda, 1], [2, 2-lambda]]
            m_txt = VGroup(
                Text(f"{3-lam_val:.1f}", font_size=20), Text("1.0", font_size=20),
                Text("2.0", font_size=20), Text(f"{2-lam_val:.1f}", font_size=20)
            ).arrange_in_grid(rows=2, buff=0.4)
            bracket_l = Text("[", font_size=40).next_to(m_txt, LEFT, buff=0.1)
            bracket_r = Text("]", font_size=40).next_to(m_txt, RIGHT, buff=0.1)
            return VGroup(bracket_l, m_txt, bracket_r).set_color(WHITE)

        curr_mat = get_mat_group(0.0)
        self.place_at_grid(curr_mat, "B5", scale_factor=1.0)
        
        # Representing slider with a visual placeholder
        slider = VGroup(
            Line(LEFT * 0.5, RIGHT * 0.5, color=WHITE),
            Dot(color=COLOR_HIGHLIGHT)
        )
        self.place_at_grid(slider, "C5", scale_factor=0.6)
        
        self.play(FadeIn(plane), FadeIn(mat_label), FadeIn(curr_mat), FadeIn(slider))
        self.wait(0.5)

        # Create collapsed state for transformation (lambda = 1.0)
        collapsed_plane = NumberPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            background_line_style={"stroke_color": COLOR_PLANE, "stroke_opacity": 0.5},
            axis_config={"stroke_color": COLOR_PLANE, "stroke_width": 2}
        ).apply_matrix([[2, 1], [2, 1]]) # A - 1I
        self.place_in_area(collapsed_plane, "D2", "F6", scale_factor=0.6)

        final_mat = get_mat_group(1.0)
        self.place_at_grid(final_mat, "B5", scale_factor=1.0)

        # Dynamic warping from lambda=0 to lambda=1
        self.play(
            Transform(plane, collapsed_plane),
            ReplacementTransform(curr_mat, final_mat),
            slider.animate.shift(RIGHT * 0.4),
            run_time=2.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(YELLOW)

        # Bold determinant equation flashing
        det_text = Text("det(A - \u03bb I) = 0", font_size=32, color=COLOR_HIGHLIGHT, weight=BOLD)
        self.place_at_grid(det_text, "B4", scale_factor=1.0)
        
        # Resulting eigenvalue display
        eigen_val = Text("\u03bb = 1.0", font_size=28, color=COLOR_HIGHLIGHT)
        self.place_at_grid(eigen_val, "A6", scale_factor=1.0)

        self.play(Flash(det_text, color=COLOR_HIGHLIGHT))
        self.play(Write(det_text))
        self.play(FadeIn(eigen_val))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(YELLOW)
        
        # Connection to null space and eigenvectors
        null_space_hint = Text("Null Space \u2192 Eigenvectors", font_size=22, color=WHITE)
        self.place_at_grid(null_space_hint, "E2", scale_factor=1.0)
        self.play(FadeIn(null_space_hint))
        
        # Highlight collapse line as visual confirmation
        collapse_indicator = Line(
            start=self.grid["D4"], 
            end=self.grid["F6"], 
            color=COLOR_HIGHLIGHT, 
            stroke_width=6
        )
        self.play(Create(collapse_indicator))
        self.wait(2)
