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
        # Setup layout with title and lecture lines
        # Lecture lines for Section 2:
        # 1. The imaginary unit i isn't just an impossible square root.
        # 2. In geometry, multiplying by i signifies a 90-degree rotation.
        # 3. It transforms flat numbers into dynamic directional vectors.
        self.setup_layout(
            "Prerequisite: The Mystery of 'i' as a Rotation",
            [
                "The imaginary unit i isn't just an impossible square root.",
                "In geometry, multiplying by i signifies a 90-degree rotation.",
                "It transforms flat numbers into dynamic directional vectors."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Replaced MathTex with Text to avoid FileNotFoundError: 'latex'
        # Display the equation 'i = √-1' (#FF69B4) in the center of the screen area.
        # Using the area A2 to B5 for the equation to keep it distinct from the rotation diagram below.
        eqn1 = Text("i = √-1", color="#FF69B4", font_size=36)
        self.place_in_area(eqn1, "A2", "B5", scale_factor=1.2)
        
        self.play(
            self.lecture[0].animate.set_color("#FF69B4"),
            Write(eqn1)
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Draw a horizontal white arrow (#FFFFFF) and rotate it 90 degrees 
        # counter-clockwise to point upwards toward an 'i' label.
        
        # We establish a local origin at E3 for the complex plane rotation.
        origin_pos = self.grid["E3"]
        
        # Initial vector pointing to '1' (at E4, which is 1 grid unit to the right)
        arrow = Arrow(
            start=origin_pos, 
            end=self.grid["E4"], 
            buff=0, 
            color=WHITE,
            stroke_width=6
        )
        
        # Labels for the positions
        label_1 = Text("1", font_size=24, color=WHITE)
        self.place_at_grid(label_1, "E5", scale_factor=1.0) # Positioned 1 unit away from the end
        
        label_i = Text("i", font_size=24, color=WHITE)
        self.place_at_grid(label_i, "C3", scale_factor=1.0) # Positioned 1 unit away from the end (D3)

        self.play(
            self.lecture[1].animate.set_color(WHITE),
            Create(arrow),
            Write(label_1)
        )
        self.wait(0.5)
        
        # Rotate the arrow 90 degrees CCW (the tip moves from E4 to D3)
        self.play(
            Rotate(arrow, angle=PI/2, about_point=origin_pos),
            Write(label_i)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Repeat the 90-degree rotation from 'i' to point left, 
        # landing on a '-1' label (#FFFFFF).
        
        # Label for the -1 position
        label_minus_1 = Text("-1", font_size=24, color=WHITE)
        self.place_at_grid(label_minus_1, "E1", scale_factor=1.0) # Positioned 1 unit away from the end (E2)
        
        self.play(
            self.lecture[2].animate.set_color(WHITE),
            Rotate(arrow, angle=PI/2, about_point=origin_pos),
            Write(label_minus_1)
        )
        self.wait(2)
