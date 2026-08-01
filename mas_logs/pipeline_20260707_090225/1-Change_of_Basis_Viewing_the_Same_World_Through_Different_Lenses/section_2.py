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
        # Configuration
        TITLE = "Prerequisite Review: The Standard Basis"
        LINES = [
            'Our standard grid uses basis vectors i and j.',
            'We reach vector V using three i and two j.',
            'Coordinates are just scalars scaling these basis vectors.'
        ]
        
        self.setup_layout(TITLE, LINES)
        
        # Dim all lines initially
        for line in self.lecture:
            line.set_color(GRAY_D)

        # Colors
        COLOR_I = "#FF0000"
        COLOR_J = "#00FF00"
        COLOR_V = "#FFFFFF"
        COLOR_GRID = "#666666"

        # === Animation for Lecture Line 1 ===
        # Highlight current line
        self.lecture[0].set_color(WHITE)
        
        # Define Coordinate System
        plane = NumberPlane(
            x_range=[-1, 5, 1],
            y_range=[-1, 4, 1],
            background_line_style={"stroke_color": COLOR_GRID, "stroke_opacity": 0.6},
            axis_config={"stroke_color": WHITE, "include_tip": True}
        )
        
        # Basis Vectors
        i_vec = Vector([1, 0], color=COLOR_I)
        j_vec = Vector([0, 1], color=COLOR_J)
        i_label = Text("i", font_size=24, slant=ITALIC, color=COLOR_I).next_to(i_vec, DOWN, buff=0.1)
        j_label = Text("j", font_size=24, slant=ITALIC, color=COLOR_J).next_to(j_vec, LEFT, buff=0.1)
        
        # Group everything for proper placement in the right area
        world = VGroup(plane, i_vec, j_vec, i_label, j_label)
        # Fix Issue 33: Start at B1 instead of A1
        self.place_in_area(world, 'B1', 'F6', scale_factor=0.8)
        
        # Animation: Fade in grid and basis vectors
        self.play(Create(plane), run_time=1.5)
        self.play(GrowArrow(i_vec), Write(i_label), run_time=1)
        self.play(GrowArrow(j_vec), Write(j_label), run_time=1)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Change lecture color
        self.lecture[0].set_color(GRAY_D)
        self.lecture[1].set_color(WHITE)
        
        # Equation: v = 3i + 2j
        equation = VGroup(
            Text("v", slant=ITALIC), 
            Text("="), 
            Text("3"), 
            Text("i", slant=ITALIC, color=COLOR_I), 
            Text("+"), 
            Text("2"), 
            Text("j", slant=ITALIC, color=COLOR_J)
        ).arrange(RIGHT, buff=0.1)
        equation.set_color(WHITE)
        equation[3].set_color(COLOR_I)
        equation[6].set_color(COLOR_J)
        
        # Fix Issue 32 & 34: Scale 0.7 at A4
        self.place_at_grid(equation, 'A4', scale_factor=0.7)
        
        self.play(Write(equation))
        self.wait(0.5)

        # Draw the path on the grid
        origin = plane.get_origin()
        unit_x = plane.get_x_unit_size()
        unit_y = plane.get_y_unit_size()
        
        # 3 units along i
        step_i = Line(origin, origin + RIGHT * 3 * unit_x, color=COLOR_I, stroke_width=4)
        # 2 units along j starting from the end of step_i
        step_j = Line(origin + RIGHT * 3 * unit_x, origin + RIGHT * 3 * unit_x + UP * 2 * unit_y, color=COLOR_J, stroke_width=4)
        
        # The resultant vector V
        v_vec = Arrow(origin, plane.coords_to_point(3, 2), buff=0, color=COLOR_V)
        v_label = Text("v = (3, 2)", font_size=24, slant=ITALIC, color=COLOR_V).next_to(v_vec.get_end(), UR, buff=0.1)
        
        self.play(Create(step_i), run_time=1)
        self.play(Create(step_j), run_time=1)
        self.play(GrowArrow(v_vec), Write(v_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Change lecture color
        self.lecture[1].set_color(GRAY_D)
        self.lecture[2].set_color(WHITE)
        
        # Highlight the scalars in the equation
        scalar_3 = equation[2]
        scalar_2 = equation[5]
        
        self.play(
            scalar_3.animate.scale(1.2).set_color(YELLOW),
            scalar_2.animate.scale(1.2).set_color(YELLOW),
        )
        self.play(
            scalar_3.animate.scale(1/1.2),
            scalar_2.animate.scale(1/1.2),
        )
        
        self.wait(2)
