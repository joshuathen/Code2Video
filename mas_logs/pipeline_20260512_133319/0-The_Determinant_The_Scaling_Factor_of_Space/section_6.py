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
        # Setup layout with specified lines
        lines = [
            'Consider a matrix where both columns are identical.',
            'The entire 2D plane collapses onto a single line.',
            "Pixel's area is squashed down to zero.",
            'A zero determinant means the transformation cannot be reversed.',
            'Information is lost when dimensions are lost.'
        ]
        self.setup_layout("The Zero Determinant: The Collapse", lines)

        # Matrix [1 1; 1 1] construction
        col1 = VGroup(Text("1", font_size=24), Text("1", font_size=24)).arrange(DOWN, buff=0.4)
        col2 = VGroup(Text("1", font_size=24), Text("1", font_size=24)).arrange(DOWN, buff=0.4)
        matrix_vals = VGroup(col1, col2).arrange(RIGHT, buff=0.6)
        bracket_l = Text("[", font_size=40)
        bracket_r = Text("]", font_size=40)
        matrix_display = VGroup(bracket_l, matrix_vals, bracket_r).arrange(RIGHT, buff=0.1)
        
        # Grid construction for the coordinate area
        # Anchor the origin at D4
        origin = self.grid["D4"]
        unit = 0.8
        
        def to_local(p):
            return origin + np.array([p[0]*unit, p[1]*unit, 0])

        # Initial state elements
        plane_grid = VGroup()
        for i in range(-2, 3):
            plane_grid.add(Line(to_local([-2, i]), to_local([2, i]), color=GRAY, stroke_opacity=0.4))
            plane_grid.add(Line(to_local([i, -2]), to_local([i, 2]), color=GRAY, stroke_opacity=0.4))
            
        i_hat = Arrow(origin, to_local([1, 0]), buff=0, color=PINK, stroke_width=4)
        j_hat = Arrow(origin, to_local([0, 1]), buff=0, color=YELLOW, stroke_width=4)
        
        # Pixel the Cat / Unit Square (Cyan #00FFFF)
        pixel_square = Polygon(
            to_local([0,0]), to_local([1,0]), to_local([1,1]), to_local([0,1]),
            fill_opacity=0.5, fill_color="#00FFFF", stroke_color="#00FFFF"
        )

        # Transformation logic: (x,y) -> (x+y, x+y)
        # Target state elements
        t_i_hat = Arrow(origin, to_local([1, 1]), buff=0, color=PINK, stroke_width=4)
        t_j_hat = Arrow(origin, to_local([1, 1]), buff=0, color=YELLOW, stroke_width=4)
        
        # The cyan square collapses to a line segment from (0,0) to (2,2)
        collapsed_pixel = Line(to_local([0,0]), to_local([2,2]), color="#00FFFF", stroke_width=6)

        # Collapsed grid lines
        collapsed_grid = VGroup()
        for i in range(-4, 5):
            # All lines collapse onto the y=x diagonal in local space
            # We represent the "density" of the collapsed plane as a thickened line
            collapsed_grid.add(Line(to_local([-2, -2]), to_local([2, 2]), color=GRAY, stroke_opacity=0.2))

        # Labels
        area_label = Text("Area = 0", font_size=24, color="#FFFFFF")
        invert_msg = Text("det(A) = 0: The transformation is not invertible", font_size=22, color="#FF0000")

        # === Animation for Lecture Line 1 ===
        # "Consider a matrix where both columns are identical."
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.place_in_area(matrix_display, 'A3', 'A4', scale_factor=0.9)
        self.add(plane_grid, pixel_square, i_hat, j_hat)
        self.play(FadeIn(matrix_display), FadeIn(plane_grid), FadeIn(pixel_square), FadeIn(i_hat), FadeIn(j_hat))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # "The entire 2D plane collapses onto a single line."
        self.play(self.lecture[0].animate.set_color(GRAY), self.lecture[1].animate.set_color("#00FFFF"))
        self.play(
            ReplacementTransform(plane_grid, collapsed_grid),
            ReplacementTransform(i_hat, t_i_hat),
            ReplacementTransform(j_hat, t_j_hat),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Pixel's area is squashed down to zero."
        self.play(self.lecture[1].animate.set_color(GRAY), self.lecture[2].animate.set_color("#00FFFF"))
        self.play(ReplacementTransform(pixel_square, collapsed_pixel))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # "A zero determinant means the transformation cannot be reversed."
        self.play(self.lecture[2].animate.set_color(GRAY), self.lecture[3].animate.set_color(RED))
        self.place_at_grid(area_label, 'E6', scale_factor=0.8)
        self.play(Write(area_label))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # "Information is lost when dimensions are lost."
        self.play(self.lecture[3].animate.set_color(GRAY), self.lecture[4].animate.set_color(RED))
        self.place_in_area(invert_msg, 'F2', 'F6', scale_factor=0.7)
        self.play(FadeIn(invert_msg))
        self.wait(2)
