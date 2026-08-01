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
        # Setup layout
        title_text = "The Matrix as a 'Landing Map'"
        lecture_lines = [
            'We only track where the basis vectors land.',
            'Watch i-hat and j-hat move to their new spots.',
            'The matrix records these two landing positions.',
            'The first column shows where i-hat landed.',
            'The second column tracks the destination of j-hat.'
        ]
        self.setup_layout(title_text, lecture_lines)

        # Colors
        I_HAT_COLOR = "#58C4DD"
        J_HAT_COLOR = "#FC6255"

        # === Animation for Lecture Line 1 ===
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=4.5,
            y_length=4.5,
            background_line_style={"stroke_opacity": 0.4}
        )
        self.place_at_grid(plane, 'D4')
        
        highlight_dot = Dot(color=YELLOW)
        self.place_at_grid(highlight_dot, 'C3', scale_factor=0.8) 

        i_hat = Vector([1, 0], color=I_HAT_COLOR).shift(plane.get_origin())
        j_hat = Vector([0, 1], color=J_HAT_COLOR).shift(plane.get_origin())
        
        i_label = Text("i", slant=ITALIC, color=I_HAT_COLOR, font_size=20)
        j_label = Text("j", slant=ITALIC, color=J_HAT_COLOR, font_size=20)
        i_label.next_to(i_hat, DOWN, buff=0.1)
        j_label.next_to(j_hat, LEFT, buff=0.1)

        self.play(Create(plane), self.lecture[0].animate.set_color(YELLOW))
        self.play(GrowArrow(i_hat), GrowArrow(j_hat), FadeIn(i_label), FadeIn(j_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        matrix_vals = [[1, -2], [1, 0]]
        
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(YELLOW),
            plane.animate.apply_matrix(matrix_vals),
            i_hat.animate.apply_matrix(matrix_vals),
            j_hat.animate.apply_matrix(matrix_vals),
            i_label.animate.move_to(plane.coords_to_point(1, 1) + RIGHT*0.2),
            j_label.animate.move_to(plane.coords_to_point(-2, 0) + UP*0.2),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        m_els = VGroup(
            Text("1", font_size=28), Text("-2", font_size=28),
            Text("1", font_size=28), Text("0", font_size=28)
        ).arrange_in_grid(rows=2, cols=2, buff=0.4)
        l_br = Text("[", font_size=45).next_to(m_els, LEFT, buff=0.1)
        r_br = Text("]", font_size=45).next_to(m_els, RIGHT, buff=0.1)
        matrix_obj = VGroup(m_els, l_br, r_br)
        
        # Define columns manually for animation
        col1 = VGroup(m_els[0], m_els[2])
        col2 = VGroup(m_els[1], m_els[3])

        self.place_at_grid(matrix_obj, 'A4', scale_factor=0.6)
        
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(YELLOW),
            FadeIn(matrix_obj)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(I_HAT_COLOR),
            col1.animate.set_color(I_HAT_COLOR),
            Indicate(i_hat, color=I_HAT_COLOR, scale_factor=1.3),
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(J_HAT_COLOR),
            col2.animate.set_color(J_HAT_COLOR),
            Indicate(j_hat, color=J_HAT_COLOR, scale_factor=1.3)
        )
        self.wait(2)
