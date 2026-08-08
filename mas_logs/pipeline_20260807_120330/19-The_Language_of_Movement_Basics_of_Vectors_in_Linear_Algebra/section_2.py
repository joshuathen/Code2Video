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
        # Section data
        title = "Prerequisite: The Coordinate Playground"
        lecture_lines = [
            "We place vectors on a 2D coordinate plane.",
            "Every vector starts at the origin (0, 0).",
            "Its destination determines its x and y components."
        ]
        self.setup_layout(title, lecture_lines)
        
        # Color definitions
        COLOR_AXES = "#555555"
        COLOR_VECTOR = "#00FFFF"
        COLOR_NOTATION = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        # "We place vectors on a 2D coordinate plane."
        self.play(self.lecture[0].animate.set_color(COLOR_AXES))
        
        # Coordinate grid (NumberPlane) positioned in the right-side area
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={
                "stroke_color": COLOR_AXES,
                "stroke_width": 1,
                "stroke_opacity": 0.3
            },
            axis_config={
                "stroke_color": COLOR_AXES,
                "include_numbers": False,
                "stroke_width": 2
            }
        )
        # Scaling to fit 8 units into 5 units of space (A1 to F6)
        # Width 8 * 0.625 = 5. Height 8 * 0.625 = 5.
        self.place_in_area(plane, 'A1', 'F6', scale_factor=0.625)
        self.play(Create(plane), run_time=1.5)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # "Every vector starts at the origin (0, 0)."
        self.play(self.lecture[1].animate.set_color(COLOR_VECTOR))
        
        # Origin and target points relative to the coordinate plane
        origin_pt = plane.get_origin()
        target_pt = plane.coords_to_point(3, 2)
        
        # Vector arrow from (0,0) to (3,2)
        vector_arrow = Arrow(
            start=origin_pt,
            end=target_pt,
            buff=0,
            color=COLOR_VECTOR,
            stroke_width=5
        )
        
        # Origin highlight
        origin_dot = Dot(origin_pt, color=COLOR_VECTOR, radius=0.04)
        
        self.play(GrowArrow(vector_arrow))
        self.play(FadeIn(origin_dot), Flash(origin_pt, color=COLOR_VECTOR))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # "Its destination determines its x and y components."
        self.play(self.lecture[2].animate.set_color(COLOR_NOTATION))
        
        # Column vector notation [3; 2]
        vector_label = MathTex(r"\begin{bmatrix} 3 \\ 2 \end{bmatrix}", color=COLOR_NOTATION)
        
        # Positioning label near the arrowhead (Grid B6 is appropriate)
        # ISSUE 17 FIX: Reduced scale factor to 0.6 to prevent cramping and screen edge proximity.
        self.place_at_grid(vector_label, 'B6', scale_factor=0.6)
        
        self.play(Write(vector_label))
        self.play(Indicate(vector_label, color=COLOR_NOTATION))
        self.wait(2)
