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
        # Define content
        title_text = "The Tall Matrix: Elevating to 3D (2x3 Transform)"
        lecture_lines = [
            "- A tall 3x2 matrix maps 2D into 3D space.",
            "- Two input coordinates become three output coordinates.",
            "- A 2D plane lifts and slants within a 3D room."
        ]
        
        # Setup the layout
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Show a 2D plane in a flat perspective with a simple 'Paper Mario' character (#FFFFFF) in the center.
        self.lecture[0].set_color(WHITE)
        
        # 2D Plane representation
        plane_rect = Rectangle(width=3.5, height=3.5, color=BLUE_E, fill_opacity=0.1)
        grid_lines = VGroup()
        # Create a grid for the "paper"
        for offset in np.linspace(-1.75, 1.75, 8):
            grid_lines.add(Line([offset, -1.75, 0], [offset, 1.75, 0], color=BLUE_D, stroke_width=1))
            grid_lines.add(Line([-1.75, offset, 0], [1.75, offset, 0], color=BLUE_D, stroke_width=1))
        
        paper_plane = VGroup(plane_rect, grid_lines)
        # Resolved Issue 25: Move paper_plane to 'C2'-'F5' and scale to 0.8
        self.place_in_area(paper_plane, "C2", "F5", scale_factor=0.8)
        
        # 'Paper Mario' character
        character = VGroup(
            Rectangle(width=0.4, height=0.6, color=WHITE, fill_opacity=1), # body
            Circle(radius=0.15, color=WHITE, fill_opacity=1).shift(UP*0.4), # head
            Line(LEFT*0.08, RIGHT*0.08, color=BLACK).shift(UP*0.45), # simple eye detail
            Line(LEFT*0.08, RIGHT*0.08, color=BLACK).shift(UP*0.35)
        ).scale(0.6)
        character.move_to(paper_plane.get_center())
        
        self.play(FadeIn(paper_plane), GrowFromCenter(character))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Display a 3x2 matrix on the left side with columns highlighted in cyan (#00FFFF)
        self.lecture[1].set_color("#00FFFF")
        
        # 3x2 Matrix
        matrix_3x2 = Matrix(
            [["1", "0"], 
             ["0", "1"], 
             ["0.5", "0.5"]],
            left_bracket="[", right_bracket="]"
        ).set_color(WHITE)
        
        # Highlight columns in cyan
        matrix_3x2.get_columns()[0].set_color("#00FFFF")
        matrix_3x2.get_columns()[1].set_color("#00FFFF")
        
        # Resolved Issue 24: Move matrix to 'A3' and scale to 0.6
        self.place_at_grid(matrix_3x2, "A3", scale_factor=0.6)
        
        self.play(Write(matrix_3x2))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Rotate the camera to a 3D perspective as the 2D plane lifts and transforms into a slanted sheet (#FFFF00) floating in 3D space.
        self.lecture[2].set_color("#FFFF00")
        
        # Transformation matrix to simulate a 3D isometric slant
        # This maps the flat 2D plane to a slanted orientation
        slant_transform = [[1, 0.5, 0], [0.3, 0.8, 0], [0, 0, 1]]
        
        # Background "room" cues to sell the 3D effect
        room_floor = Line(self.grid["F1"], self.grid["F6"], color=GRAY, stroke_width=2).set_opacity(0.5)
        room_wall = Line(self.grid["F1"], self.grid["A1"], color=GRAY, stroke_width=2).set_opacity(0.5)
        
        self.play(
            Create(room_floor),
            Create(room_wall),
            paper_plane.animate.apply_matrix(slant_transform).set_color("#FFFF00").shift(UP*1.0),
            character.animate.apply_matrix(slant_transform).shift(UP*1.0),
            run_time=2,
            rate_func=smooth
        )
        self.wait(2)
