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
        self.setup_layout(
            "The Determinant: Scaling the Area",
            [
                "Determinants measure how much space is scaled.",
                "Look at the area of a single unit square.",
                "The transformation stretches or squishes this area.",
                "This area scaling factor is the determinant.",
                "A determinant of zero collapses space into a line."
            ]
        )

        # Local coordinate system helper
        # Origin at D2
        origin_pos = self.grid["D2"]
        def to_s(p):
            # Scale 0.5 to keep everything within A1-F6 right-side area
            return origin_pos + np.array([p[0]*0.5, p[1]*0.5, 0])

        # === Animation for Lecture Line 1 ===
        # Determinants measure how much space is scaled.
        self.play(self.lecture[0].animate.set_color(YELLOW))
        
        grid_lines = VGroup()
        for i in range(-2, 4):
            grid_lines.add(Line(to_s([i, -2, 0]), to_s([i, 3, 0]), color="#FFFFFF", stroke_opacity=0.3))
        for i in range(-2, 4):
            grid_lines.add(Line(to_s([-2, i, 0]), to_s([3, i, 0]), color="#FFFFFF", stroke_opacity=0.3))
        
        self.play(Create(grid_lines))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Look at the area of a single unit square.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(GOLD)
        )
        
        # Gold 1x1 unit square
        unit_square = Polygon(
            to_s([0, 0, 0]), to_s([1, 0, 0]), to_s([1, 1, 0]), to_s([0, 1, 0]),
            color="#FFD700", fill_opacity=0.5, fill_color="#FFD700"
        )
        area_label_1 = Text("Area = 1", font_size=20, color="#FFD700")
        self.place_at_grid(area_label_1, "C2", scale_factor=0.8)
        area_label_1.shift(UP * 0.3)

        self.play(Create(unit_square))
        self.play(Write(area_label_1))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The transformation stretches or squishes this area.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(BLUE)
        )
        
        # Matrix M1 = [[2, 1], [0, 1.5]] -> Area = 3
        def m1(p):
            return [2*p[0] + p[1], 1.5*p[1], 0]

        para_poly = Polygon(
            to_s(m1([0, 0, 0])), to_s(m1([1, 0, 0])), to_s(m1([1, 1, 0])), to_s(m1([0, 1, 0])),
            color=BLUE, fill_opacity=0.5, fill_color=BLUE
        )
        
        m1_grid = VGroup()
        for i in range(-2, 4):
            m1_grid.add(Line(to_s(m1([i, -2, 0])), to_s(m1([i, 3, 0])), color=BLUE, stroke_opacity=0.3))
        for i in range(-2, 4):
            m1_grid.add(Line(to_s(m1([-2, i, 0])), to_s(m1([3, i, 0])), color=BLUE, stroke_opacity=0.3))

        self.play(
            Transform(unit_square, para_poly),
            Transform(grid_lines, m1_grid),
            FadeOut(area_label_1),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # This area scaling factor is the determinant.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(YELLOW)
        )
        
        det_text = Text("det(A) = 3", font_size=32, color=YELLOW)
        self.place_at_grid(det_text, "B4")
        
        # Issue 36 fix: place at B2
        area_label_3 = Text("Area = 3", font_size=20, color=YELLOW)
        self.place_at_grid(area_label_3, "B2", scale_factor=0.8)

        self.play(Write(det_text), Write(area_label_3))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        # A determinant of zero collapses space into a line.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(RED)
        )
        
        # Matrix M2 = [[1, 1], [1, 1]] -> Area = 0
        def m2(p):
            return [p[0] + p[1], p[0] + p[1], 0]

        flat_poly = Polygon(
            to_s(m2([0, 0, 0])), to_s(m2([1, 0, 0])), to_s(m2([1, 1, 0])), to_s(m2([0, 1, 0])),
            color=RED, fill_opacity=1.0, fill_color=RED
        )
        
        m2_grid = VGroup()
        for i in range(-2, 4):
            m2_grid.add(Line(to_s(m2([i, -2, 0])), to_s(m2([i, 3, 0])), color=RED, stroke_opacity=0.6))
        for i in range(-2, 4):
            m2_grid.add(Line(to_s(m2([-2, i, 0])), to_s(m2([3, i, 0])), color=RED, stroke_opacity=0.6))

        # Issue 37 fix: place at A4
        det_zero = Text("det(A) = 0", font_size=32, color=RED)
        self.place_at_grid(det_zero, "A4")

        self.play(
            Transform(unit_square, flat_poly),
            Transform(grid_lines, m2_grid),
            Transform(det_text, det_zero),
            FadeOut(area_label_3),
            run_time=2
        )
        self.play(Indicate(det_text, color=RED, scale_factor=1.3))
        self.wait(2)
