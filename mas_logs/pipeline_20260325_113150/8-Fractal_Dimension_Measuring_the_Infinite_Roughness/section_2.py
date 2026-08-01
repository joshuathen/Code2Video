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
        # Initialize layout with title and lecture lines
        self.setup_layout("Prerequisite: Euclidean Scaling Laws", [
            "Standard shapes follow simple scaling rules.",
            "Double a square's side, four copies fit.",
            "For these, dimension is a whole number."
        ])

        # Colors for each dimension
        COLOR_1D = "#FFFFFF"  # White
        COLOR_2D = "#00FFFF"  # Cyan
        COLOR_3D = "#FF00FF"  # Magenta

        # === Animation for Lecture Line 1 ===
        # Standard shapes follow simple scaling rules. (1D Line Example)
        self.play(self.lecture[0].animate.set_color(COLOR_1D))
        
        line_unit = Line(LEFT*0.3, RIGHT*0.3, color=COLOR_1D, stroke_width=4)
        arrow_1 = Arrow(LEFT*0.5, RIGHT*0.5, color=WHITE, buff=0.1)
        
        # Scaling factor S=2, Number of parts N=2
        line_scaled_part1 = Line(LEFT*0.3, RIGHT*0.3, color=COLOR_1D, stroke_width=4)
        line_scaled_part2 = Line(LEFT*0.3, RIGHT*0.3, color=COLOR_1D, stroke_width=4)
        line_scaled = VGroup(line_scaled_part1, line_scaled_part2).arrange(RIGHT, buff=0.1)
        
        line_demo = VGroup(line_unit, arrow_1, line_scaled).arrange(RIGHT, buff=0.5)
        # Resolved Issue 48: Area changed to A1-B4
        self.place_in_area(line_demo, 'A1', 'B4', scale_factor=1.2)
        
        label_1d = Text("1D: N = S^1", font_size=24, color=COLOR_1D)
        # Resolved Issue 47 & 49: Area changed to B5-B6 for better fit and alignment
        self.place_in_area(label_1d, 'B5', 'B6', scale_factor=0.8)
        
        self.play(Create(line_unit))
        self.play(GrowArrow(arrow_1))
        self.play(TransformFromCopy(line_unit, line_scaled))
        self.play(Write(label_1d))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Double a square's side, four copies fit. (2D Square Example)
        self.play(self.lecture[1].animate.set_color(COLOR_2D))
        
        sq_unit = Square(side_length=0.6, color=COLOR_2D)
        arrow_2 = Arrow(LEFT*0.5, RIGHT*0.5, color=WHITE, buff=0.1)
        
        # Scaling factor S=2, Number of parts N=2^2 = 4
        sq_scaled = VGroup(*[
            Square(side_length=0.6, color=COLOR_2D) for _ in range(4)
        ]).arrange_in_grid(rows=2, cols=2, buff=0.05)
        
        sq_demo = VGroup(sq_unit, arrow_2, sq_scaled).arrange(RIGHT, buff=0.5)
        # Resolved Issue 48: Area changed to C1-D4
        self.place_in_area(sq_demo, 'C1', 'D4', scale_factor=1.1)
        
        label_2d = Text("2D: N = S^2", font_size=24, color=COLOR_2D)
        # Resolved Issue 47 & 49: Area changed to D5-D6
        self.place_in_area(label_2d, 'D5', 'D6', scale_factor=0.8)
        
        self.play(Create(sq_unit))
        self.play(GrowArrow(arrow_2))
        self.play(TransformFromCopy(sq_unit, sq_scaled))
        self.play(Write(label_2d))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # For these, dimension is a whole number. (3D Cube Example)
        self.play(self.lecture[2].animate.set_color(COLOR_3D))
        
        # Helper to draw a simple isometric cube projection
        def get_cube_proj(color, side=0.4):
            s1 = Square(side_length=side, color=color, stroke_width=2)
            s2 = s1.copy().shift(UP*side*0.3 + RIGHT*side*0.3)
            l1 = Line(s1.get_corner(UL), s2.get_corner(UL), color=color, stroke_width=2)
            l2 = Line(s1.get_corner(UR), s2.get_corner(UR), color=color, stroke_width=2)
            l3 = Line(s1.get_corner(DL), s2.get_corner(DL), color=color, stroke_width=2)
            l4 = Line(s1.get_corner(DR), s2.get_corner(DR), color=color, stroke_width=2)
            return VGroup(s1, s2, l1, l2, l3, l4)

        cube_unit = get_cube_proj(COLOR_3D)
        arrow_3 = Arrow(LEFT*0.5, RIGHT*0.5, color=WHITE, buff=0.1)
        
        # Scaling factor S=2, Number of parts N=2^3 = 8
        cubes_scaled = VGroup()
        # Create a 2x2x2 arrangement of small cubes
        for z in [1, 0]: # Back layer then front layer
            for r in [1, 0]: # Top to bottom
                for c in [0, 1]: # Left to right
                    cb = get_cube_proj(COLOR_3D, side=0.35)
                    # Shift based on row, column and layer (z)
                    cb.shift(c*0.4*RIGHT + r*0.4*UP + z*(0.15*UP + 0.15*RIGHT))
                    cubes_scaled.add(cb)
        
        cube_demo = VGroup(cube_unit, arrow_3, cubes_scaled).arrange(RIGHT, buff=0.5)
        # Resolved Issue 48: Area changed to E1-F4
        self.place_in_area(cube_demo, 'E1', 'F4', scale_factor=1.0)
        
        label_3d = Text("3D: N = S^3", font_size=24, color=COLOR_3D)
        # Resolved Issue 47 & 49: Area changed to F5-F6
        self.place_in_area(label_3d, 'F5', 'F6', scale_factor=0.8)
        
        self.play(Create(cube_unit))
        self.play(GrowArrow(arrow_3))
        self.play(TransformFromCopy(cube_unit, cubes_scaled))
        self.play(Write(label_3d))
        self.wait(2)
