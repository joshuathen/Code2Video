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
        # Initialize the scene layout with title and lecture script
        self.setup_layout(
            "Prerequisite: The Hidden Power of 'i'",
            [
                "In math, i is the root of negative one.",
                "Geometrically, multiplying by i creates a ninety-degree rotation.",
                "It turns a horizontal real number into a vertical one."
            ]
        )

        # Initial lecture text color (dimmed to allow for highlights)
        self.lecture.set_color(GREY_B)

        # === Animation for Lecture Line 1 ===
        # Use ComplexPlane for the visualization
        # Defined with clear ranges for the unit arrows
        plane = ComplexPlane(
            x_range=[-2.2, 2.2, 1],
            y_range=[-2.2, 2.2, 1],
            background_line_style={
                "stroke_color": BLUE_D,
                "stroke_width": 2,
                "stroke_opacity": 0.4
            }
        )
        
        # Position the plane in the designated grid area (Issue 28, 29, 42)
        # Using A2 to F6 and scale 0.9 to balance the layout and improve legibility.
        self.place_in_area(plane, "A2", "F6", scale_factor=0.9)
        
        origin_pt = plane.n2p(0)
        pos_1 = plane.n2p(1)
        
        # Create the initial real-axis arrow
        arrow = Arrow(origin_pt, pos_1, buff=0, color=WHITE, stroke_width=6)
        
        # Manual label using Text (Issue 42: No MathTex)
        # Positioned within 1 grid unit of the point (L003)
        label_1 = Text("1", color=WHITE, font_size=32).next_to(pos_1, DOWN + RIGHT, buff=0.1)
        
        self.play(
            Create(plane),
            GrowArrow(arrow),
            Write(label_1),
            self.lecture[0].animate.set_color(WHITE)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Rotate the arrow 90 degrees to point at 'i'
        pos_i = plane.n2p(1j)
        label_i = Text("i", color="#58C4DD", font_size=32).next_to(pos_i, UP + RIGHT, buff=0.1)
        
        self.play(
            Rotate(arrow, angle=PI/2, about_point=origin_pt),
            arrow.animate.set_color("#58C4DD"),
            Write(label_i),
            self.lecture[1].animate.set_color("#58C4DD")
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Rotate the arrow another 90 degrees to point at -1
        pos_neg_1 = plane.n2p(-1)
        label_neg_1 = Text("-1", color="#FF0000", font_size=32).next_to(pos_neg_1, DOWN + LEFT, buff=0.1)
        
        self.play(
            Rotate(arrow, angle=PI/2, about_point=origin_pt),
            arrow.animate.set_color("#FF0000"),
            Write(label_neg_1),
            self.lecture[2].animate.set_color("#FF0000")
        )
        self.wait(2)
