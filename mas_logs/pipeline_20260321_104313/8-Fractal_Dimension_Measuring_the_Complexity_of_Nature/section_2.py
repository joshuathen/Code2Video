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
        # Setup the layout with the title and lecture lines
        self.setup_layout(
            "Prerequisite: The Logic of Integer Dimensions",
            [
                "Standard dimensions follow a simple scaling rule.",
                "Halving a square's sides creates four smaller squares.",
                "For a cube, halving sides creates eight pieces."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Standard dimensions follow a simple scaling rule.
        # Highlight first line and display formula 'N = s^D' in white.
        # Using Text instead of MathTex to avoid 'latex' FileNotFoundError
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        formula = Text("N = s^D", color="#FFFFFF")
        self.place_in_area(formula, "A2", "A5", scale_factor=1.2)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Halving a square's sides creates four smaller squares.
        # Highlight second line and draw a green square divided into 4.
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        # Big square
        big_square = Square(side_length=1.5, color="#00FF00")
        self.place_in_area(big_square, "B1", "D3", scale_factor=1.0)
        self.play(Create(big_square))
        
        # Division lines for the square (visualizing halving sides)
        h_line = Line(big_square.get_left(), big_square.get_right(), color="#00FF00")
        v_line = Line(big_square.get_top(), big_square.get_bottom(), color="#00FF00")
        self.play(Create(h_line), Create(v_line))
        
        # Label for square
        # Using Text instead of MathTex to avoid 'latex' FileNotFoundError
        label_sq = Text("s=2, N=4 -> D=2", color="#00FF00", font_size=24)
        self.place_at_grid(label_sq, "E2")
        self.play(Write(label_sq))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # For a cube, halving sides creates eight pieces.
        # Highlight third line and draw a blue wireframe cube divided into 8.
        self.play(self.lecture[2].animate.set_color("#0000FF"))
        
        # Helper logic for a wireframe cube
        def get_wireframe_cube(side=1.2, color="#0000FF"):
            offset = 0.3
            front = Square(side_length=side, color=color)
            back = Square(side_length=side, color=color).shift(RIGHT*offset + UP*offset)
            l_ul = Line(front.get_corner(UL), back.get_corner(UL), color=color)
            l_ur = Line(front.get_corner(UR), back.get_corner(UR), color=color)
            l_dl = Line(front.get_corner(DL), back.get_corner(DL), color=color)
            l_dr = Line(front.get_corner(DR), back.get_corner(DR), color=color)
            return VGroup(front, back, l_ul, l_ur, l_dl, l_dr)

        cube = get_wireframe_cube()
        self.place_in_area(cube, "B4", "D6", scale_factor=1.0)
        self.play(Create(cube))
        
        # Division lines for the wireframe cube (visualizing 8 internal pieces)
        f_mid_h = Line(cube[0].get_left(), cube[0].get_right(), color="#0000FF")
        f_mid_v = Line(cube[0].get_top(), cube[0].get_bottom(), color="#0000FF")
        b_mid_h = Line(cube[1].get_left(), cube[1].get_right(), color="#0000FF")
        b_mid_v = Line(cube[1].get_top(), cube[1].get_bottom(), color="#0000FF")
        
        # Midpoints of depth lines
        m_ul = cube[2].point_from_proportion(0.5)
        m_ur = cube[3].point_from_proportion(0.5)
        m_dl = cube[4].point_from_proportion(0.5)
        m_dr = cube[5].point_from_proportion(0.5)
        
        d_mid_v1 = Line(m_ul, m_dl, color="#0000FF")
        d_mid_v2 = Line(m_ur, m_dr, color="#0000FF")
        d_mid_h1 = Line(m_ul, m_ur, color="#0000FF")
        d_mid_h2 = Line(m_dl, m_dr, color="#0000FF")
        
        cube_divs = VGroup(f_mid_h, f_mid_v, b_mid_h, b_mid_v, d_mid_v1, d_mid_v2, d_mid_h1, d_mid_h2)
        self.play(Create(cube_divs))
        
        # Label for cube
        # Using Text instead of MathTex to avoid 'latex' FileNotFoundError
        label_cube = Text("s=2, N=8 -> D=3", color="#0000FF", font_size=24)
        self.place_at_grid(label_cube, "E5")
        self.play(Write(label_cube))
        self.wait(2)
