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
        # Define lecture lines
        lecture_lines = [
            "Imagine Pixel defines his own set of basis vectors.",
            "These new vectors must be linearly independent.",
            "They form a tilted and stretched coordinate grid.",
            "Let's look at Pixel's world versus our own.",
            "The same point now has two different labels."
        ]
        
        self.setup_layout("Defining a New Basis", lecture_lines)

        # Colors
        V1_COLOR = "#FFFF00"  # Yellow
        V2_COLOR = "#00FFFF"  # Cyan
        HIGHLIGHT_COLOR = "#FF00FF" # Magenta
        STD_GRID_COLOR = "#444444"

        # Setup coordinate system area
        # Fix Issue 39: Adjusted area and scale to prevent obstruction
        plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={
                "stroke_color": STD_GRID_COLOR,
                "stroke_width": 1,
                "stroke_opacity": 0.5
            }
        )
        self.place_in_area(plane, 'B3', 'F6', scale_factor=0.5)
        
        # Vectors v1 and v2
        origin = plane.c2p(0, 0)
        v1_end = plane.c2p(2, 1)
        v2_end = plane.c2p(-1, 1)
        
        v1 = Vector(v1_end - origin, color=V1_COLOR).shift(origin)
        v2 = Vector(v2_end - origin, color=V2_COLOR).shift(origin)
        
        v1_label = MathTex(r"\vec{v}_1", color=V1_COLOR, font_size=24)
        v2_label = MathTex(r"\vec{v}_2", color=V2_COLOR, font_size=24)
        
        # Position labels near the heads of the vectors
        v1_label.next_to(v1.get_end(), RIGHT, buff=0.1)
        v2_label.next_to(v2.get_end(), LEFT, buff=0.1)

        # Basis label
        # Fix Issue 40: Adjusted position and scale for better visibility
        basis_label = MathTex(r"B = \{ \vec{v}_1, \vec{v}_2 \}", font_size=32)
        self.place_at_grid(basis_label, 'A5', scale_factor=0.8)

        # Skewed plane
        matrix = [[2, -1], [1, 1]]
        skewed_plane = NumberPlane(
            x_range=[-4, 4, 1],
            y_range=[-4, 4, 1],
            background_line_style={
                "stroke_color": HIGHLIGHT_COLOR,
                "stroke_width": 2,
                "stroke_opacity": 0.6
            }
        ).apply_matrix(matrix)
        skewed_plane.move_to(origin)

        # Dashed lines for independence check
        line1 = DashedLine(plane.c2p(-4, -2), plane.c2p(4, 2), color=V1_COLOR, stroke_opacity=0.3)
        line2 = DashedLine(plane.c2p(4, -4), plane.c2p(-4, 4), color=V2_COLOR, stroke_opacity=0.3)

        # Unit cell
        unit_cell = Polygon(
            origin,
            v1.get_end(),
            v1.get_end() + (v2.get_end() - origin),
            v2.get_end(),
            fill_opacity=0.3,
            fill_color=HIGHLIGHT_COLOR,
            stroke_color=HIGHLIGHT_COLOR
        )

        # Point and labels
        target_coords_standard = [1, 2]
        target_point_pos = plane.c2p(*target_coords_standard)
        dot = Dot(target_point_pos, color=WHITE)
        
        label_std = MathTex(r"(1, 2)_{\text{std}}", color=WHITE, font_size=24)
        label_b = MathTex(r"[1, 1]_B", color=V2_COLOR, font_size=24)
        label_std.next_to(dot, UR, buff=0.1)
        label_b.next_to(label_std, DOWN, aligned_edge=LEFT, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(V1_COLOR))
        self.add(plane)
        self.play(GrowArrow(v1), Write(v1_label))
        self.play(self.lecture[0].animate.set_color(V2_COLOR))
        self.play(GrowArrow(v2), Write(v2_label))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color(WHITE))
        self.play(Create(line1), Create(line2))
        self.play(Write(basis_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color(HIGHLIGHT_COLOR))
        self.play(
            FadeOut(plane),
            FadeOut(line1),
            FadeOut(line2),
            Create(skewed_plane),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color(V1_COLOR))
        self.play(FadeIn(unit_cell))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color(V2_COLOR))
        self.play(Create(dot))
        self.play(Write(label_std))
        self.play(Write(label_b))
        self.wait(2)
