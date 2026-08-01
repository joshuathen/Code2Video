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
        # Fetching data from storyboard/outline
        title = "Scalars vs. Vectors: The Intuition"
        lines = [
            "Scalars represent quantity alone, like speed or temperature.",
            "Vectors add direction, showing where that quantity is headed.",
            "Visualize a vector as an arrow in space.",
            "The arrow's length represents the magnitude or intensity.",
            "The arrowhead points exactly where the vector is going."
        ]
        self.setup_layout(title, lines)

        # Colors
        SCALAR_COLOR = "#FFD700"  # Gold
        VECTOR_COLOR = "#1E90FF"  # DodgerBlue
        DIRECTION_COLOR = "#FF4500"  # OrangeRed

        # === Animation for Lecture Line 1 ===
        # Scalars represent quantity alone, like speed or temperature.
        self.play(self.lecture[0].animate.set_color(SCALAR_COLOR))
        
        scalar_val = Text("5 km/h", font_size=36, color=SCALAR_COLOR)
        # Issue 19 Fix: Position at A3, scale 1.2
        self.place_at_grid(scalar_val, 'A3', scale_factor=1.2)
        self.play(FadeIn(scalar_val))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Vectors add direction, showing where that quantity is headed.
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color(VECTOR_COLOR)
        )
        
        # Start vector at C2 to avoid top scalar
        v_start = self.grid["D2"]
        v_end = self.grid["D4"]
        vector = Arrow(v_start, v_end, color=VECTOR_COLOR, buff=0)
        
        self.play(GrowArrow(vector))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Visualize a vector as an arrow in space.
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color(VECTOR_COLOR)
        )
        
        # Subtle coordinate axes
        axis_h = Line(self.grid["E1"], self.grid["E6"], color=GRAY, stroke_width=1).set_opacity(0.3)
        axis_v = Line(self.grid["F2"], self.grid["B2"], color=GRAY, stroke_width=1).set_opacity(0.3)
        self.play(Create(axis_h), Create(axis_v))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # The arrow's length represents the magnitude or intensity.
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            self.lecture[3].animate.set_color(VECTOR_COLOR)
        )
        
        # Increase arrow length
        new_end = self.grid["D6"]
        self.play(vector.animate.put_start_and_end_on(v_start, new_end), run_time=1.5)
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # The arrowhead points exactly where the vector is going.
        self.play(
            self.lecture[3].animate.set_color(WHITE),
            self.lecture[4].animate.set_color(DIRECTION_COLOR)
        )
        
        # Rotate and Pulse to emphasize direction
        self.play(Rotate(vector, angle=PI/4, about_point=v_start), run_time=1.2)
        
        # Pulse arrowhead (whole vector for visual simplicity)
        self.play(vector.animate.set_color(DIRECTION_COLOR), run_time=0.4)
        self.play(vector.animate.set_color(VECTOR_COLOR), run_time=0.4)
        self.play(vector.animate.set_color(DIRECTION_COLOR), run_time=0.4)
        self.play(vector.animate.set_color(VECTOR_COLOR), run_time=0.4)
        
        self.wait(2)
