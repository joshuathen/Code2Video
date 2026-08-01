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
        # Initial layout setup with title and lecture lines
        self.setup_layout("Prerequisite: Iteration and Recursive Logic", [
            "Space-filling curves rely on simple, repeating recursive rules.",
            "We start with a base shape, like this U-curve.",
            "This rule repeats at smaller scales to increase complexity."
        ])

        # === Animation for Lecture Line 1 ===
        # Show a square grid frame in #FFFFFF with a 'Rule' label appearing at the top.
        # Area A1 to F6 covers the full right workspace.
        frame = Square(side_length=4.5, color="#FFFFFF", stroke_width=2)
        self.place_in_area(frame, "A1", "F6")

        rule_label = Text("Rule", font_size=28, color="#FFFFFF")
        # Placing the label centered at the top of the area
        self.place_in_area(rule_label, "A3", "A4")

        self.play(
            self.lecture[0].animate.set_color("#FFFFFF"),
            Create(frame),
            Write(rule_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A bright #FFFF00 U-shaped path connects four points in the grid center.
        # This represents the base recursive step (Hilbert curve order 0).
        u_base_points = [
            np.array([-1, 1, 0]),
            np.array([-1, -1, 0]),
            np.array([1, -1, 0]),
            np.array([1, 1, 0])
        ]
        u_base = VMobject().set_points_as_corners(u_base_points).set_color("#FFFF00")
        
        # Place it inside the square frame
        self.place_in_area(u_base, "B2", "E5", scale_factor=0.8)
        
        self.play(
            self.lecture[1].animate.set_color("#FFFF00"),
            Create(u_base)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The U-shape splits into four smaller, rotated U-shapes in #FF8C00.
        # This represents the first level of recursion (Hilbert curve order 1).
        
        def get_small_u():
            pts = [
                np.array([-0.5, 0.5, 0]),
                np.array([-0.5, -0.5, 0]),
                np.array([0.5, -0.5, 0]),
                np.array([0.5, 0.5, 0])
            ]
            return VMobject().set_points_as_corners(pts)

        # Define the four oriented sub-shapes
        u_tl = get_small_u()
        u_tr = get_small_u()
        u_bl = get_small_u().rotate(PI/2)
        u_br = get_small_u().rotate(-PI/2)

        # Group them to create the split-visual effect
        recursive_u_group = VGroup(
            VGroup(u_tl, u_tr).arrange(RIGHT, buff=0.8),
            VGroup(u_bl, u_br).arrange(RIGHT, buff=0.8)
        ).arrange(DOWN, buff=0.8).set_color("#FF8C00")

        # Position the new group in the same workspace area
        self.place_in_area(recursive_u_group, "B2", "E5", scale_factor=0.6)

        self.play(
            self.lecture[2].animate.set_color("#FF8C00"),
            Transform(u_base, recursive_u_group)
        )
        self.wait(2)
