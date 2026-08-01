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

class TeachingScene(ThreeDScene):
    def setup_layout(self, title_text, lecture_lines):
        # BASE
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color="#FFFFFF").to_edge(UP)
        self.add_fixed_in_frame_mobjects(self.title)

        # Left-side lecture content (bullets with "-")
        lecture_texts = [Text(line, font_size=22, color="#FFFFFF") for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT).scale(0.8)
        self.lecture.to_edge(LEFT, buff=0.2)
        self.add_fixed_in_frame_mobjects(self.lecture)

        # Define fine-grained animation grid (6x6 grid on right side)
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
        # Data from storyboard
        title_text = "Tall Matrices: Moving Up (2D to 3D)"
        lecture_lines = [
            "Tall matrices map 2D inputs to 3D outputs.",
            "Two input coordinates become three output coordinates.",
            "This embeds a 2D plane into 3D space.",
            "The matrix columns define where the grid lands.",
            "Our flat world now lives in a larger room."
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        color_matrix = "#FFFFFF"
        color_x = "#FF0000"
        color_y = "#00FF00"
        color_z = "#0000FF"
        color_plane_2d = "#333333"
        color_v1 = "#FFFF00"
        color_v2 = "#00FFFF"
        color_span = "#555555"
        color_square = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(color_matrix)
        
        # Matrix at B4 (Resolved Issue 26)
        matrix = MathTex(
            r"A = \begin{bmatrix} v_{1x} & v_{2x} \\ v_{1y} & v_{2y} \\ v_{1z} & v_{2z} \end{bmatrix}",
            color=color_matrix
        )
        self.place_at_grid(matrix, "B4", scale_factor=0.9)
        self.add(matrix) # In 3D space to "shift perspective" later
        
        # 3D Axes at D4-F6 (Resolved Issue 25)
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-3, 3, 1],
            x_axis_config={"color": color_x},
            y_axis_config={"color": color_y},
            z_axis_config={"color": color_z},
            axis_config={"include_tip": True}
        )
        self.place_in_area(axes, "D4", "F6", scale_factor=0.5)
        
        self.play(Write(matrix))
        self.play(FadeIn(axes))
        self.wait(1.0)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(color_y)
        
        # 2D plane on XY
        xy_plane = NumberPlane(
            x_range=[-2.5, 2.5, 1],
            y_range=[-2.5, 2.5, 1],
            background_line_style={"stroke_color": color_plane_2d, "stroke_opacity": 0.6}
        )
        xy_plane.move_to(axes.get_center())
        xy_plane.scale(0.5)
        
        self.play(Create(xy_plane))
        self.wait(0.5)
        
        # Shift perspective
        self.move_camera(phi=75 * DEGREES, theta=-45 * DEGREES, run_time=2)
        self.wait(1.0)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(color_v1)
        
        # Column vectors
        v1_val = np.array([1, 1, 1.5])
        v2_val = np.array([1.5, -0.5, 0.5])
        
        origin = axes.get_origin()
        v1_end = axes.c2p(*v1_val)
        v2_end = axes.c2p(*v2_val)
        
        v1_arrow = Arrow(origin, v1_end, buff=0, color=color_v1, stroke_width=4)
        v2_arrow = Arrow(origin, v2_end, buff=0, color=color_v2, stroke_width=4)
        
        v1_label = MathTex(r"\vec{v}_1", color=color_v1).scale(0.7)
        v2_label = MathTex(r"\vec{v}_2", color=color_v2).scale(0.7)
        
        # Position labels near the vector tips (Proximity Rule)
        v1_label.move_to(v1_end + np.array([0, 0, 0.3]))
        v2_label.move_to(v2_end + np.array([0, 0, 0.3]))
        
        self.play(GrowArrow(v1_arrow), Write(v1_label))
        self.play(GrowArrow(v2_arrow), Write(v2_label))
        self.wait(1.0)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(color_v2)
        
        # Translucent plane (span)
        # Defining 4 points on the plane to form a large enough polygon
        p1 = axes.c2p(*(2*v1_val + 2*v2_val))
        p2 = axes.c2p(*(2*v1_val - 2*v2_val))
        p3 = axes.c2p(*(-2*v1_val - 2*v2_val))
        p4 = axes.c2p(*(-2*v1_val + 2*v2_val))
        
        span_plane = Polygon(p1, p2, p3, p4, color=color_span, fill_opacity=0.3, stroke_width=0)
        
        self.play(FadeIn(span_plane))
        self.wait(1.0)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(color_square)
        
        # 2D square transformed to 3D
        square_corners_2d = [
            np.array([0, 0, 0]),
            np.array([1, 0, 0]),
            np.array([1, 1, 0]),
            np.array([0, 1, 0])
        ]
        square_2d = Polygon(*[axes.c2p(*p) for p in square_corners_2d], color=color_square, fill_opacity=0.3)
        
        # Transformed corners using linear combination of v1 and v2
        square_corners_3d = [
            axes.c2p(0, 0, 0),
            axes.c2p(*v1_val),
            axes.c2p(*(v1_val + v2_val)),
            axes.c2p(*v2_val)
        ]
        square_3d = Polygon(*square_corners_3d, color=color_square, fill_opacity=0.5)
        
        self.play(Create(square_2d))
        self.wait(0.5)
        self.play(Transform(square_2d, square_3d))
        self.wait(2.0)
