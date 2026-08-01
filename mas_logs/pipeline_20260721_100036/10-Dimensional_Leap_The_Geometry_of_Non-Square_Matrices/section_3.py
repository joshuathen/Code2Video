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
        title_str = "Tall Matrices: Upgrading Dimensions (2D to 3D)"
        lines_str = [
            "A 3x2 matrix maps 2D input to 3D output.",
            "Two input basis vectors land in a 3D world.",
            "This creates a flat plane slicing through 3D space.",
            "The transformation embeds the lower dimension into higher space.",
            "Notice how the 2D grid lifts into the room."
        ]
        
        self.setup_layout(title_str, lines_str)
        
        # Colors from storyboard
        c1 = "#FFFFFF" # White
        c2 = "#808080" # Gray
        c3 = "#00FF00" # Green
        c4 = "#FFFF00" # Yellow
        c5 = "#FF00FF" # Magenta

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(c1)
        plane_2d = NumberPlane(
            x_range=[-2, 2, 1], y_range=[-2, 2, 1], 
            background_line_style={"stroke_color": BLUE_E, "stroke_width": 1},
            axis_config={"stroke_width": 2, "include_tip": True}
        ).scale(0.5)
        
        char_head = Circle(radius=0.1, color=c1, fill_opacity=0.8)
        char_body = Triangle(color=c1, fill_opacity=0.5).scale(0.15).next_to(char_head, DOWN, buff=0)
        char = VGroup(char_head, char_body).move_to(plane_2d.c2p(1, 1))
        
        group_2d = VGroup(plane_2d, char)
        # Apply Fix for Issue 24
        self.place_in_area(group_2d, "A2", "C3", scale_factor=0.8)
        
        self.play(FadeIn(group_2d))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(c2)
        
        v_x = np.array([0.8, -0.4, 0])
        v_y = np.array([0.8, 0.4, 0])
        v_z = np.array([0, 1.0, 0])
        
        origin_3d = self.grid["E5"]
        axes_3d = VGroup(
            Arrow(origin_3d, origin_3d + v_x * 1.5, color=c2, buff=0, stroke_width=2),
            Arrow(origin_3d, origin_3d + v_y * 1.5, color=c2, buff=0, stroke_width=2),
            Arrow(origin_3d, origin_3d + v_z * 1.5, color=c2, buff=0, stroke_width=2)
        )
        axes_label = Text("3D Space", font_size=16, color=c2).next_to(axes_3d, DOWN, buff=0.1)
        
        basis_i = Arrow(origin_3d, origin_3d + v_x, color=RED, buff=0, stroke_width=4)
        basis_j = Arrow(origin_3d, origin_3d + v_y, color=GREEN, buff=0, stroke_width=4)
        
        self.play(Create(axes_3d), Write(axes_label))
        self.play(GrowArrow(basis_i), GrowArrow(basis_j))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(c3)
        
        tilted_plane = Polygon(
            origin_3d, 
            origin_3d + v_x * 2.5, 
            origin_3d + v_x * 2.5 + v_y * 2.5, 
            origin_3d + v_y * 2.5,
            color=c3, fill_opacity=0.3, stroke_width=2
        ).shift(-(v_x * 1.25 + v_y * 1.25))
        
        char_3d = char.copy().scale(0.8)
        char_3d.move_to(origin_3d + v_x * 0.5 + v_y * 0.5)
        
        self.play(
            group_2d.animate.scale(0.6).next_to(axes_3d, LEFT, buff=0.8),
            FadeIn(tilted_plane),
            TransformFromCopy(char, char_3d)
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color(c4)
        
        vol_label = Text("Planar Output (No Volume)", font_size=18, color=c4)
        # Apply Fix for Issue 22
        self.place_in_area(vol_label, "D3", "D5", scale_factor=0.8)
        
        thickness_line = Line(origin_3d, origin_3d + v_z * 0.8, color=c4, stroke_width=2)
        zero_label = Text("h = 0", font_size=14, color=c4).next_to(thickness_line, RIGHT, buff=0.1)
        
        self.play(Write(vol_label))
        self.play(Create(thickness_line), FadeIn(zero_label))
        self.play(Indicate(tilted_plane, color=c4))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color(c5)
        
        # Create matrix and highlight its columns
        matrix_mobject = Matrix(
            [["a_{11}", "a_{12}"], 
             ["a_{21}", "a_{22}"], 
             ["a_{31}", "a_{32}"]],
            left_bracket="[",
            right_bracket="]"
        ).set_color(c5)
        matrix_label = MathTex("A =", color=c5).next_to(matrix_mobject, LEFT)
        matrix = VGroup(matrix_label, matrix_mobject)
        
        # Apply Fix for Issue 23
        self.place_in_area(matrix, "A4", "C6", scale_factor=0.7)
        
        entries = matrix_mobject.get_entries()
        # Matrix is row-major: 0,1 | 2,3 | 4,5
        col1 = VGroup(entries[0], entries[2], entries[4])
        col2 = VGroup(entries[1], entries[3], entries[5])
        
        self.play(FadeIn(matrix))
        self.play(
            col1.animate.set_color(RED),
            col2.animate.set_color(GREEN)
        )
        self.play(
            Indicate(basis_i, color=RED),
            Indicate(col1, color=RED)
        )
        self.play(
            Indicate(basis_j, color=GREEN),
            Indicate(col2, color=GREEN)
        )
        self.wait(2)
