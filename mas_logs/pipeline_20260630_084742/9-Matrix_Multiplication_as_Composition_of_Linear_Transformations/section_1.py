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
        lecture_lines = [
            "Matrices aren't just boxes; they are transformations.",
            "Think of a matrix as a verb moving space.",
            "Meet Leo the Lion on our coordinate grid.",
            "This rotation matrix tilts Leo ninety degrees.",
            "The entire space transforms as a single unit."
        ]
        self.setup_layout("The Big Idea: Functions in Space", lecture_lines)

        # Coordinate Grid
        plane = NumberPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            background_line_style={"stroke_color": "#444444", "stroke_width": 2},
            axis_config={"stroke_color": "#444444"}
        )

        # Leo Character Construction
        leo_head = Circle(radius=0.5, color="#D2B48C", fill_opacity=1)
        leo_mane = Annulus(inner_radius=0.5, outer_radius=0.7, color="#8B4513", fill_opacity=1)
        leo_eye_l = Dot(point=LEFT*0.2 + UP*0.1, color=BLACK).scale(0.5)
        leo_eye_r = Dot(point=RIGHT*0.2 + UP*0.1, color=BLACK).scale(0.5)
        leo_smile = Arc(radius=0.2, start_angle=PI, angle=PI, color=BLACK).shift(DOWN*0.1)
        leo = VGroup(leo_mane, leo_head, leo_eye_l, leo_eye_r, leo_smile).scale(0.6)

        # Container for the grid and Leo to move together
        space_group = VGroup(plane, leo)
        # Fix Issue 29: Move to C3-F6 and scale 0.8
        self.place_in_area(space_group, 'C3', 'F6', scale_factor=0.8)

        # Matrix object
        matrix_elements = VGroup(
            VGroup(Text("0", font_size=24), Text("-1", font_size=24)).arrange(RIGHT, buff=0.6),
            VGroup(Text("1", font_size=24), Text("0", font_size=24)).arrange(RIGHT, buff=0.6)
        ).arrange(DOWN, buff=0.4)
        
        l_bracket = Text("[", font_size=40).scale([1, 2, 1])
        r_bracket = Text("]", font_size=40).scale([1, 2, 1])
        l_bracket.next_to(matrix_elements, LEFT, buff=0.1)
        r_bracket.next_to(matrix_elements, RIGHT, buff=0.1)
        matrix_group = VGroup(l_bracket, matrix_elements, r_bracket).set_color("#00FF00")
        
        # Fix Issue 28 & 30: Move to A5-A6 and scale 0.7
        self.place_in_area(matrix_group, 'A5', 'A6', scale_factor=0.7)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(plane))
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        self.play(
            plane.animate.set_stroke(color=WHITE, width=4),
            self.lecture[1].animate.set_color(WHITE),
            run_time=0.5
        )
        self.play(
            plane.animate.set_stroke(color="#444444", width=2),
            run_time=0.5
        )

        # === Animation for Lecture Line 3 ===
        self.play(
            FadeIn(leo),
            self.lecture[2].animate.set_color("#D2B48C")
        )
        self.play(
            leo.animate.set_color(YELLOW),
            run_time=0.3
        )
        self.play(
            leo.animate.set_color(WHITE),
            run_time=0.3
        )
        # Restore Leo's characteristic colors for now
        leo[0].set_color("#8B4513")
        leo[1].set_color("#D2B48C")
        leo[2:5].set_color(BLACK)
        self.wait(0.5)

        # === Animation for Lecture Line 4 ===
        self.play(
            Write(matrix_group),
            self.lecture[3].animate.set_color("#00FF00")
        )
        self.play(
            space_group.animate.rotate(90 * DEGREES),
            run_time=2
        )

        # === Animation for Lecture Line 5 ===
        self.play(
            leo.animate.scale(1.2),
            self.lecture[4].animate.set_color(WHITE)
        )
        self.play(
            leo.animate.scale(1/1.2).set_color(WHITE),
            run_time=1
        )
        self.wait(2)
