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

class Section5Scene(TeachingScene):
    def construct(self):
        # === Data setup ===
        lecture_lines = [
            "We calculate x as a ratio of areas.",
            "x equals the new area divided by original.",
            "This is the determinant ratio of modified matrices.",
            "Similarly, find y by swapping v2 with b.",
            "Cramer's rule solves variables using determinants."
        ]
        self.setup_layout("Deriving the Ratio", lecture_lines)

        COLOR_X = "#00FF00" # Green
        COLOR_Y = "#00FFFF" # Cyan
        COLOR_V1 = BLUE
        COLOR_V2 = RED
        COLOR_B = YELLOW

        # Setup coordinate system - positioned in bottom-right to avoid derivation rows
        plane = NumberPlane(
            x_range=[0, 14, 2], y_range=[0, 4, 1],
            x_length=4, y_length=2.5,
            background_line_style={"stroke_opacity": 0.4}
        )
        self.place_in_area(plane, 'E2', 'F6', scale_factor=0.8)
        
        # Vectors
        v1_vec = Arrow(plane.c2p(0, 0), plane.c2p(3, 0), buff=0, color=COLOR_V1)
        v2_vec = Arrow(plane.c2p(0, 0), plane.c2p(1, 1), buff=0, color=COLOR_V2)
        b_vec = Arrow(plane.c2p(0, 0), plane.c2p(29/3, 5/3), buff=0, color=COLOR_B)
        
        v1_label = MathTex(r"\vec{v}_1", color=COLOR_V1, font_size=20).next_to(v1_vec.get_end(), DOWN, buff=0.1)
        v2_label = MathTex(r"\vec{v}_2", color=COLOR_V2, font_size=20).next_to(v2_vec.get_end(), LEFT, buff=0.1)
        b_label = MathTex(r"\vec{b}", color=COLOR_B, font_size=20).next_to(b_vec.get_end(), UP, buff=0.1)

        # Parallelograms
        poly_orig = Polygon(
            plane.c2p(0,0), plane.c2p(3,0), plane.c2p(4,1), plane.c2p(1,1),
            color=WHITE, fill_opacity=0.3, stroke_width=2
        )
        # Parallelogram for x (using b and v2)
        poly_x = Polygon(
            plane.c2p(0,0), plane.c2p(29/3, 5/3), plane.c2p(32/3, 8/3), plane.c2p(1,1),
            color=COLOR_X, fill_opacity=0.3, stroke_width=2
        )
        # Parallelogram for y (using v1 and b)
        poly_y = Polygon(
            plane.c2p(0,0), plane.c2p(3,0), plane.c2p(38/3, 5/3), plane.c2p(29/3, 5/3),
            color=COLOR_Y, fill_opacity=0.3, stroke_width=2
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(WHITE)
        formula_x_ratio = MathTex(r"x = \frac{\text{Area}(\vec{b}, \vec{v}_2)}{\text{Area}(\vec{v}_1, \vec{v}_2)}", color=WHITE)
        self.place_at_grid(formula_x_ratio, 'B2', scale_factor=0.7)
        
        self.play(Write(formula_x_ratio))
        self.play(Create(plane), Create(v1_vec), Create(v2_vec), Create(b_vec), Write(v1_label), Write(v2_label), Write(b_label))
        self.play(Create(poly_orig), Create(poly_x))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_X)
        calc_x = MathTex(r"x = \frac{8}{3}", color=COLOR_X)
        self.place_at_grid(calc_x, 'B6', scale_factor=0.8)
        
        self.play(Write(calc_x))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(WHITE)
        formula_x_det = MathTex(r"x = \frac{\det(\vec{b}, \vec{v}_2)}{\det(\vec{v}_1, \vec{v}_2)}", color=WHITE)
        self.place_at_grid(formula_x_det, 'B4', scale_factor=0.7)
        
        self.play(Write(formula_x_det))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(COLOR_Y)
        # Swap visualizations to highlight y
        self.play(
            FadeOut(poly_x),
            Create(poly_y)
        )
        
        formula_y_det = MathTex(r"y = \frac{\det(\vec{v}_1, \vec{b})}{\det(\vec{v}_1, \vec{v}_2)}", color=COLOR_Y)
        self.place_at_grid(formula_y_det, 'D4', scale_factor=0.7)
        
        calc_y = MathTex(r"y = \frac{5}{3}", color=COLOR_Y)
        self.place_at_grid(calc_y, 'D6', scale_factor=0.8)
        
        self.play(Write(formula_y_det))
        self.play(Write(calc_y))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(WHITE)
        final_coords = MathTex(r"(x, y) = \left( \frac{8}{3}, \frac{5}{3} \right)", color=WHITE)
        self.place_at_grid(final_coords, 'A4', scale_factor=1.0)
        
        self.play(Write(final_coords))
        self.play(Indicate(final_coords))
        self.wait(2)
