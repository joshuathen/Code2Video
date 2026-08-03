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
        title_text = "Prerequisite: Scalar vs. Vector"
        lecture_lines = [
            "Scalars represent size or magnitude alone.",
            "Vectors include both magnitude and direction.",
            "Think of a dot versus an arrow."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Display a white dot (#FFFFFF) labeled "Scalar: 2 cm/s".
        self.play(self.lecture[0].animate.set_color(WHITE))
        
        dot = Dot(color=WHITE)
        self.place_at_grid(dot, "C3", scale_factor=1.2)
        
        scalar_label = Text("Scalar: 2 cm/s", font_size=20, color=WHITE)
        self.place_at_grid(scalar_label, "B3", scale_factor=0.8)
        
        self.play(
            FadeIn(dot),
            Write(scalar_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Transform the dot into a green arrow (#00FF00) pointing right labeled "Vector: 2 cm/s Right".
        self.play(self.lecture[1].animate.set_color("#00FF00"))
        
        arrow = Arrow(start=LEFT, end=RIGHT, color="#00FF00", buff=0)
        self.place_at_grid(arrow, "C3", scale_factor=1.0)
        
        vector_label = Text("Vector: 2 cm/s Right", font_size=20, color="#00FF00")
        self.place_in_area(vector_label, "B3", "B5", scale_factor=0.8)
        
        self.play(
            ReplacementTransform(dot, arrow),
            ReplacementTransform(scalar_label, vector_label)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Scale the green arrow (#00FF00) to double its length to show increased magnitude.
        self.play(self.lecture[2].animate.set_color("#00FF00"))
        
        # We update the label too for context
        magnitude_label = Text("Magnitude increases", font_size=18, color="#00FF00")
        self.place_in_area(magnitude_label, "D3", "D5", scale_factor=0.8)

        self.play(
            arrow.animate.scale(2),
            FadeIn(magnitude_label)
        )
        self.wait(2)
