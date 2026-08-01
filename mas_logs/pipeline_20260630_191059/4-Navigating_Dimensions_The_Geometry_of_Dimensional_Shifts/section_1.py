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

class Section1Scene(TeachingScene):
    def construct(self):
        # Initial setup using outline and storyboard data
        title_text = "The Foundation: Building the Ladder"
        lecture_lines = [
            "Dimensions grow by moving perpendicular to existing space.",
            "Zero dimensions is just a single stationary point.",
            "Move that point to create a one-dimensional line.",
            "Shift the line sideways to form a two-dimensional square.",
            "Pull the square upward to build a three-dimensional cube."
        ]
        self.setup_layout(title_text, lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Dimensions grow by moving perpendicular to existing space.
        self.lecture[0].set_color(WHITE)
        
        # Define key points on the grid
        origin_pt = self.grid['D2']
        x_end = self.grid['D5']
        y_end = self.grid['A2']
        z_end = self.grid['F1']
        
        # Create axes arrows
        x_axis = Arrow(origin_pt, x_end, color="#FF0000", buff=0, stroke_width=4)
        y_axis = Arrow(origin_pt, y_end, color="#00FF00", buff=0, stroke_width=4)
        z_axis = Arrow(origin_pt, z_end, color="#0000FF", buff=0, stroke_width=4)
        
        # Label axes using grid placement
        x_lbl = Text("X", font_size=20, color="#FF0000")
        self.place_at_grid(x_lbl, 'D6', scale_factor=0.8)
        
        y_lbl = Text("Y", font_size=20, color="#00FF00")
        self.place_at_grid(y_lbl, 'A1', scale_factor=0.8)
        
        z_lbl = Text("Z", font_size=20, color="#0000FF")
        self.place_at_grid(z_lbl, 'F2', scale_factor=0.8)
        
        axes_grp = VGroup(x_axis, y_axis, z_axis, x_lbl, y_lbl, z_lbl)
        self.play(Create(axes_grp), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Zero dimensions is just a single stationary point.
        self.lecture[1].set_color(WHITE)
        
        point_0d = Dot(point=origin_pt, color=WHITE, radius=0.1)
        point_label = Text("0D Point", font_size=24, color=WHITE)
        self.place_at_grid(point_label, 'C1', scale_factor=0.8)
        
        self.play(FadeIn(point_0d), Write(point_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Move that point to create a one-dimensional line.
        self.lecture[2].set_color("#FFFF00")
        
        line_1d = Line(origin_pt, x_end, color="#FFFF00", stroke_width=6)
        line_label = Text("1D Line", font_size=24, color="#FFFF00")
        # Fix for Issue 24: Reposition label to avoid overlap
        self.place_at_grid(line_label, 'E5', scale_factor=0.7)
        
        self.play(
            ReplacementTransform(point_0d.copy(), line_1d),
            FadeIn(line_label),
            FadeOut(point_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Shift the line sideways to form a two-dimensional square.
        self.lecture[3].set_color("#00FF00")
        
        # Define square corners on grid: D2, D5, A5, A2
        square_corners = [self.grid['D2'], self.grid['D5'], self.grid['A5'], self.grid['A2']]
        square_2d = Polygon(*square_corners, color="#00FF00", fill_opacity=0.3, stroke_width=4)
        square_label = Text("2D Square", font_size=24, color="#00FF00")
        # Fix for Issue 23: Reposition label to avoid clutter
        self.place_at_grid(square_label, 'A4', scale_factor=0.7)
        
        self.play(
            ReplacementTransform(line_1d.copy(), square_2d),
            FadeIn(square_label),
            FadeOut(line_label),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Pull the square upward to build a three-dimensional cube.
        self.lecture[4].set_color("#00FFFF")
        
        # Back face: A2, A5, D5, D2
        # Front face shifted toward viewer: C1, C4, F4, F1
        back_corners = [self.grid['A2'], self.grid['A5'], self.grid['D5'], self.grid['D2']]
        front_corners = [self.grid['C1'], self.grid['C4'], self.grid['F4'], self.grid['F1']]
        
        cube_back = Polygon(*back_corners, color="#00FFFF", stroke_width=2)
        cube_front = Polygon(*front_corners, color="#00FFFF", fill_opacity=0.4, stroke_width=4)
        cube_conns = VGroup(*[Line(back_corners[i], front_corners[i], color="#00FFFF", stroke_width=2) for i in range(4)])
        cube_grp = VGroup(cube_back, cube_conns, cube_front)
        
        cube_label = Text("3D Cube", font_size=24, color="#00FFFF")
        # Fix for Issue 25: Reposition label to avoid Z-axis interference
        self.place_at_grid(cube_label, 'F5', scale_factor=0.7)
        
        self.play(
            ReplacementTransform(square_2d.copy(), cube_grp),
            FadeIn(cube_label),
            FadeOut(square_label),
            run_time=2
        )
        self.wait(3)
