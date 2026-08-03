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
        # Setup the layout with section title and lecture lines
        self.setup_layout("Prerequisite: Determinant as Area", [
            "Determinants measure the area of a parallelogram.",
            "Two vectors form the sides of this shape.",
            "If the area is zero, the grid collapses."
        ])

        # === Animation for Lecture Line 1 ===
        # Show unit square formed by [1, 0] and [0, 1] in #FFFFFF.
        self.lecture[0].set_color(WHITE)
        
        # Create and position a coordinate system
        plane = NumberPlane(
            x_range=[-1, 5], 
            y_range=[-1, 5], 
            x_length=4.5, 
            y_length=4.5,
            background_line_style={"stroke_color": GREY, "stroke_opacity": 0.4}
        )
        # Fix Issue 34: Coordinate plane scale 0.8
        self.place_in_area(plane, 'A1', 'F6', scale_factor=0.8)
        
        # Unit square defined at the origin of the plane
        square = Polygon(
            plane.c2p(0, 0), plane.c2p(1, 0), plane.c2p(1, 1), plane.c2p(0, 1), 
            color=WHITE, fill_opacity=0.2
        )
        v1_unit = Arrow(plane.c2p(0, 0), plane.c2p(1, 0), buff=0, color=WHITE, stroke_width=4)
        v2_unit = Arrow(plane.c2p(0, 0), plane.c2p(0, 1), buff=0, color=WHITE, stroke_width=4)
        
        self.play(
            Create(plane),
            DrawBorderThenFill(square),
            GrowArrow(v1_unit),
            GrowArrow(v2_unit),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transform square into parallelogram using v1 and v2 in #00FFFF.
        self.lecture[1].set_color("#00FFFF")
        
        # Target coordinates for vectors v1 and v2
        v1_coords = [2, 1, 0]
        v2_coords = [1, 3, 0]
        v_sum = [3, 4, 0]

        parallelogram = Polygon(
            plane.c2p(0, 0), 
            plane.c2p(*v1_coords), 
            plane.c2p(*v_sum), 
            plane.c2p(*v2_coords), 
            color="#00FFFF", 
            fill_opacity=0.5
        )
        v1_arrow = Arrow(plane.c2p(0, 0), plane.c2p(*v1_coords), buff=0, color="#00FFFF", stroke_width=4)
        v2_arrow = Arrow(plane.c2p(0, 0), plane.c2p(*v2_coords), buff=0, color="#00FFFF", stroke_width=4)
        
        # Fix Issue 35: det(A) label at B5, scale 0.7
        det_label = MathTex(r"\text{det}(A)", color="#00FFFF")
        self.place_at_grid(det_label, 'B5', scale_factor=0.7)
        
        self.play(
            Transform(square, parallelogram),
            Transform(v1_unit, v1_arrow),
            Transform(v2_unit, v2_arrow),
            Write(det_label),
            run_time=2
        )
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # If the area is zero, the grid collapses.
        self.lecture[2].set_color("#FF5555")
        
        # Collapsed state: v2 is collinear with v1 (e.g., v2 = 0.5 * v1)
        v2_col_coords = [1, 0.5, 0]
        v_col_sum = [3, 1.5, 0]
        
        collapsed_poly = Polygon(
            plane.c2p(0, 0), 
            plane.c2p(*v1_coords), 
            plane.c2p(*v_col_sum), 
            plane.c2p(*v2_col_coords), 
            color="#FF5555", 
            fill_opacity=0.8
        )
        v2_col_arrow = Arrow(plane.c2p(0, 0), plane.c2p(*v2_col_coords), buff=0, color="#FF5555", stroke_width=4)
        
        # Fix Issue 36: det(A) = 0 label at C6, scale 0.7
        zero_det_label = MathTex(r"\text{det}(A) = 0", color="#FF5555")
        self.place_at_grid(zero_det_label, 'C6', scale_factor=0.7)

        self.play(
            Transform(square, collapsed_poly),
            Transform(v2_unit, v2_col_arrow),
            v1_unit.animate.set_color("#FF5555"),
            Transform(det_label, zero_det_label),
            run_time=2
        )
        self.wait(2)
