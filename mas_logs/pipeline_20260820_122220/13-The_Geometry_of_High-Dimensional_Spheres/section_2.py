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
        self.setup_layout("Mathematical Framework: The General Equation", 
                          ["The n-dimensional sphere follows a standard equation.", 
                           "Sum the squares of all coordinates.", 
                           "This total equals the radius squared."])
        
        # === Animation for Lecture Line 1 ===
        # Equation: x_1^2 + x_2^2 + ... + x_n^2 = r^2
        # Applied fix: Balanced placement using B2-C5 range, scale_factor 0.8
        eq = MathTex("x_1^2 + x_2^2 + \\dots + x_n^2 = r^2", font_size=42)
        self.place_in_area(eq, 'B2', 'C5', scale_factor=0.8)
        self.play(Write(eq))
        self.lecture[0].set_color("#3498DB")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight summation part
        self.lecture[1].set_color("#F1C40F")
        self.play(Indicate(eq[0][0:8])) # Highlight sum part
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Point to radius squared
        self.lecture[2].set_color("#2ECC71") # Updated per storyboard instruction
        self.play(Flash(eq[0][-4:]))
        self.wait(2)
