from manim import *
import numpy as np

# === Base TeachingScene Class ===
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

# === Section 4 Scene ===
class Section4Scene(TeachingScene):
    def construct(self):
        # 1. Setup Layout
        title_text = "The Determinant: Measuring the 'Stretch'"
        lecture_lines = [
            "The determinant measures how much transformations scale areas.",
            "Watch this unit square stretch after a transformation.",
            "An area tripled means the determinant is three.",
            "The determinant value represents the scaling factor.",
            "A determinant of zero squishes space into a line."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        h_color = "#FF6600" # Highlight color
        square_color = "#FFFF00"
        det_color = "#00FF00"
        grid_color = "#555555"
        zero_color = "#FF0000"
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(h_color))
        
        # Create a coordinate plane
        plane = NumberPlane(
            x_range=[-2, 4, 1],
            y_range=[-2, 4, 1],
            x_length=4,
            y_length=4,
            background_line_style={
                "stroke_color": grid_color,
                "stroke_opacity": 0.6
            }
        )
        self.place_in_area(plane, "C2", "E5")
        plane_origin = plane.coords_to_point(0, 0)

        self.play(Create(plane))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(h_color)
        )
        
        # Unit square
        unit_square = Polygon(
            plane.coords_to_point(0, 0),
            plane.coords_to_point(1, 0),
            plane.coords_to_point(1, 1),
            plane.coords_to_point(0, 1),
            color=square_color,
            fill_opacity=0.5,
            stroke_width=2
        )
        
        # Area label
        area_label = Text("Area = 1", font_size=18, color=square_color)
        self.place_at_grid(area_label, "C4")

        self.play(FadeIn(unit_square), Write(area_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(h_color)
        )
        
        # Transformation matrix (det=3): A = [[2, 1], [1, 2]]
        matrix_a = [[2, 1], [1, 2]]
        
        self.play(
            plane.animate.apply_matrix(matrix_a, about_point=plane_origin),
            unit_square.animate.apply_matrix(matrix_a, about_point=plane_origin),
            FadeOut(area_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(h_color)
        )
        
        # Fix Issue 23: Move det_label to B5, scale 0.8
        det_label = Text("det(A) = 3", font_size=24, color=det_color)
        self.place_at_grid(det_label, "B5", scale_factor=0.8)
        
        # Fix Issue 24: Move new_area_label to D5, scale 0.8
        new_area_label = Text("Area = 3", font_size=18, color=det_color)
        self.place_at_grid(new_area_label, "D5", scale_factor=0.8)

        self.play(Write(det_label), FadeIn(new_area_label))
        self.wait(2)

        # === Animation for Lecture Line 5 ===
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(h_color)
        )
        
        # Singular matrix transformation
        # Create a target for the transformation
        plane_zero_ref = NumberPlane(
            x_range=[-2, 4, 1],
            y_range=[-2, 4, 1],
            x_length=4,
            y_length=4,
            background_line_style={
                "stroke_color": WHITE,
                "stroke_opacity": 0.8
            }
        )
        plane_zero_ref.apply_matrix([[1, 1], [1, 1]], about_point=plane_zero_ref.coords_to_point(0, 0))
        self.place_in_area(plane_zero_ref, "C2", "E5")
        
        square_zero = Polygon(
            plane_zero_ref.coords_to_point(0, 0),
            plane_zero_ref.coords_to_point(1, 0),
            plane_zero_ref.coords_to_point(1, 1),
            plane_zero_ref.coords_to_point(0, 1),
            color=square_color,
            fill_opacity=0.5,
            stroke_width=2
        )
        
        # Fix Issue 25: Move det_label_zero to B5, scale 0.8
        det_label_zero = Text("det(A) = 0", font_size=24, color=zero_color)
        self.place_at_grid(det_label_zero, "B5", scale_factor=0.8)

        self.play(
            Transform(plane, plane_zero_ref),
            Transform(unit_square, square_zero),
            Transform(det_label, det_label_zero),
            FadeOut(new_area_label)
        )
        self.wait(2)
