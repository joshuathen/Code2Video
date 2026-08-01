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
        # Setup layout
        title = "Prerequisite: The Qubit and Superposition"
        lines = [
            "A qubit is the quantum version of a bit.",
            "It exists in a superposition of two states simultaneously.",
            "We visualize this as a vector in 2D space."
        ]
        self.setup_layout(title, lines)

        # Colors
        axis_color = "#FFFFFF"
        label0_color = "#00BFFF"
        label1_color = "#FF6347"
        vector_color = "#FFFF00"
        equation_color = "#FFFFFF"

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(YELLOW)
        
        # Draw white axes
        origin = self.grid["E2"]
        x_end = self.grid["E6"]
        y_end = self.grid["A2"]
        
        axis_x = Arrow(origin, x_end, color=axis_color, buff=0, stroke_width=4)
        axis_y = Arrow(origin, y_end, color=axis_color, buff=0, stroke_width=4)
        
        # Fixed: Replaced MathTex with Text to avoid FileNotFoundError for 'latex'
        label_0 = Text("|0⟩", color=label0_color, font_size=24)
        # Resolved Issue 38: Move label_0 from F6 to E6
        self.place_at_grid(label_0, "E6", scale_factor=0.8)
        
        label_1 = Text("|1⟩", color=label1_color, font_size=24)
        # Resolved Issue 37: Move label_1 from A1 to B1
        self.place_at_grid(label_1, "B1", scale_factor=0.8)

        self.play(Create(axis_x), Create(axis_y))
        self.play(Write(label_0), Write(label_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(YELLOW)
        
        # Create a unit vector
        vec_len = np.linalg.norm(self.grid["E6"] - origin)
        # Vector points to |0> initially
        psi_vector = Arrow(origin, origin + RIGHT * vec_len, color=vector_color, buff=0, stroke_width=6)
        
        # Sweeping animation
        self.play(Create(psi_vector))
        
        # Rotate from |0> towards |1> (90 degrees)
        self.play(
            Rotate(psi_vector, angle=PI/2, about_point=origin, rate_func=smooth),
            run_time=2
        )
        # Rotate back to a middle state (superposition)
        self.play(
            Rotate(psi_vector, angle=-PI/4, about_point=origin, rate_func=smooth),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(YELLOW)
        
        # Fixed: Replaced MathTex with Text using unicode characters for math
        equation = Text("|ψ⟩ = α|0⟩ + β|1⟩", color=equation_color, font_size=24)
        # Resolved Issue 39: Move equation from A4-A6 to B4-B6
        self.place_in_area(equation, "B4", "B6", scale_factor=0.9)
        
        self.play(Write(equation))
        self.wait(2)

        # Reset final line color
        self.play(self.lecture[2].animate.set_color(WHITE))
        self.wait(1)
