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
        # Setup title and lecture lines
        lecture_lines = [
            "Start with a vector in the standard coordinate system.",
            "Now, introduce a new, tilted grid called Basis B.",
            "Basis B uses its own unique set of unit vectors.",
            "The vector remains fixed as we swap underlying grids.",
            "Notice how its coordinates change in this new perspective."
        ]
        self.setup_layout("The Core Problem: The Tilted Grid", lecture_lines)

        # Pre-creating decoration objects
        circle = Circle(radius=0.3, color=BLUE)
        square = Square(side_length=0.4, color=RED)
        triangle = Triangle().scale(0.3).set_color(GREEN)
        
        self.place_at_grid(circle, 'A2', scale_factor=0.7)
        self.place_at_grid(triangle, 'A4', scale_factor=0.7)
        self.place_at_grid(square, 'A6', scale_factor=0.7)
        self.add(circle, square, triangle)

        # Standard Grid Setup
        std_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": "#444444", "stroke_width": 2},
            axis_config={"include_tip": True, "stroke_color": "#444444"}
        )
        self.place_in_area(std_grid, 'B2', 'F6', scale_factor=0.9)
        grid_origin = std_grid.get_origin()

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        # Vector v = (2, 2) in standard grid
        v_end = std_grid.c2p(2, 2)
        v_vec = Arrow(grid_origin, v_end, buff=0, color=WHITE, stroke_width=4)
        # Using Text instead of MathTex to avoid FileNotFoundError: 'latex'
        v_label_std = Text("v = [2, 2] S", font_size=24, color=WHITE)
        v_label_std.next_to(v_end, UR, buff=0.1)

        self.play(Create(std_grid), run_time=1)
        self.play(GrowArrow(v_vec), Write(v_label_std))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color(YELLOW))

        # Tilted Basis B: b1=(1,1), b2=(-1,1)
        tilted_grid = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={"stroke_color": "#AA00FF", "stroke_opacity": 0.5, "stroke_width": 1},
            axis_config={"stroke_color": "#AA00FF", "stroke_width": 2}
        )
        self.place_in_area(tilted_grid, 'B2', 'F6', scale_factor=0.9)
        tilted_grid.apply_matrix([[1, -1], [1, 1]])

        self.play(FadeIn(tilted_grid), std_grid.animate.set_stroke(opacity=0.2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color(YELLOW))

        # Basis vectors b1 and b2
        b1_end = std_grid.c2p(1, 1)
        b2_end = std_grid.c2p(-1, 1)
        b1_vec = Arrow(grid_origin, b1_end, buff=0, color="#00FF00", stroke_width=4)
        b2_vec = Arrow(grid_origin, b2_end, buff=0, color="#FF0000", stroke_width=4)
        # Using Text instead of MathTex to avoid dependency error
        b1_label = Text("b1", font_size=20, color="#00FF00").next_to(b1_end, DR, buff=0.05)
        b2_label = Text("b2", font_size=20, color="#FF0000").next_to(b2_end, DL, buff=0.05)

        self.play(GrowArrow(b1_vec), Write(b1_label))
        self.play(GrowArrow(b2_vec), Write(b2_label))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(YELLOW))

        self.play(
            std_grid.animate.set_stroke(opacity=0.0),
            tilted_grid.animate.set_stroke(opacity=1.0),
            run_time=1
        )
        self.wait(1)
        self.play(
            std_grid.animate.set_stroke(opacity=0.2),
            tilted_grid.animate.set_stroke(opacity=0.5),
            run_time=1
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(YELLOW))

        # Coordinates change to (2,0) in Basis B
        v_label_tilted = Text("v = [2, 0] B", font_size=24, color="#AA00FF")
        v_label_tilted.next_to(v_label_std, DOWN, aligned_edge=LEFT, buff=0.2)

        self.play(Write(v_label_tilted))
        self.wait(2)

        # Cleanup
        self.play(self.lecture[4].animate.set_color(WHITE))
        self
