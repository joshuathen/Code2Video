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
        # Mandatory layout setup
        self.setup_layout("Prerequisite Knowledge: Jordan Curves", [
            "These loops are formally known as Jordan curves.",
            "They are simple, closed, and never cross themselves.",
            "We can stretch them into any \"blob\" shape."
        ])

        # Colors
        PURPLE_HEX = "#BB86FC"  # Visible purple
        WHITE_HEX = "#FFFFFF"
        GREEN_HEX = "#00FF00"

        # === Animation for Lecture Line 1 ===
        # Create a purple loop (Jordan curve)
        # A parametric function that forms a simple closed loop
        loop = ParametricFunction(
            lambda t: np.array([
                1.4 * np.cos(t) + 0.15 * np.sin(2 * t),
                1.4 * np.sin(t) + 0.1 * np.cos(3 * t),
                0
            ]),
            t_range=[0, TAU],
            color=PURPLE_HEX
        )
        # Position the loop in the central right area
        self.place_in_area(loop, "B2", "E5", scale_factor=0.8)
        
        label = Text("Jordan Curve", font_size=24, color=WHITE_HEX)
        # Position label above the loop using the grid
        self.place_at_grid(label, "A3", scale_factor=1.0)

        self.play(
            self.lecture[0].animate.set_color(PURPLE_HEX),
            Create(loop),
            Write(label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # A white dot travels around the loop to show it's simple and closed
        dot = Dot(color=WHITE_HEX)
        # Align dot start point
        dot.move_to(loop.point_from_proportion(0))
        
        self.play(
            self.lecture[1].animate.set_color(WHITE_HEX),
            FadeIn(dot)
        )
        self.play(
            MoveAlongPath(dot, loop),
            run_time=4,
            rate_func=linear
        )
        self.play(FadeOut(dot))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Smoothly morph the loop into a green jagged blob
        points = []
        num_vertices = 50
        for i in range(num_vertices):
            angle = i * TAU / num_vertices
            # Add jaggedness via overlapping sine waves
            radius = 1.4 + 0.3 * np.sin(12 * angle) + 0.2 * np.cos(7 * angle)
            points.append([radius * np.cos(angle), radius * np.sin(angle), 0])
        
        jagged_blob = Polygon(*points, color=GREEN_HEX)
        # Align the blob to the same area as the loop for a clean Transform
        self.place_in_area(jagged_blob, "B2", "E5", scale_factor=0.8)

        self.play(
            self.lecture[2].animate.set_color(GREEN_HEX),
            Transform(loop, jagged_blob)
        )
        self.wait(2)
