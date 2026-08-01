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
        # Mandatory call to setup layout with title and lecture lines
        self.setup_layout("The Hook: Shapey the Snail’s Trail", [
            "A snail crawls in a messy, closed loop.",
            "Can we find four points forming a square?",
            "This is the classic \"Square Peg Problem\"."
        ])

        # === Animation for Lecture Line 1 ===
        # Highlight the corresponding lecture line with color
        self.play(self.lecture[0].animate.set_color("#A020F0"))
        
        # Create a complex, non-self-intersecting loop in purple (#A020F0)
        # We define a series of points on the right-side grid and smooth them
        loop_points = [
            self.grid["B2"], 
            self.grid["A3"], 
            self.grid["B5"], 
            self.grid["D6"], 
            self.grid["F5"], 
            self.grid["E3"], 
            self.grid["F1"], 
            self.grid["D1"],
            self.grid["B2"] # Ensure it closes
        ]
        loop = VMobject().set_points_smoothly(loop_points)
        loop.set_color("#A020F0")
        
        # Draw the loop
        self.play(Create(loop), run_time=3)
        self.wait(0.5)

        # === Animation for Lecture Line 2 ===
        # Highlight the corresponding lecture line with color
        self.play(self.lecture[1].animate.set_color("#FFFF00"))
        
        # Create a yellow (#FFFF00) square frame
        square_frame = Square(side_length=1.5, color="#FFFF00")
        # Place it initially in the visualization area
        self.place_at_grid(square_frame, "C3")
        
        # Animate the square frame moving around, "trying" to touch the loop corners
        # We'll do a few translations and rotations to simulate a search
        self.play(square_frame.animate.move_to(self.grid["B4"]).rotate(PI/4), run_time=1.5)
        self.play(square_frame.animate.move_to(self.grid["D4"]).rotate(-PI/3), run_time=1.5)
        self.play(square_frame.animate.move_to(self.grid["C5"]).rotate(PI/8), run_time=1.5)
        
        self.wait(0.5)

        # === Animation for Lecture Line 3 ===
        # Highlight the corresponding lecture line with color
        self.play(self.lecture[2].animate.set_color("#FFFFFF"))
        
        # Create the text for the problem name
        problem_label = Text("Square Peg Problem", font_size=32, color="#FFFFFF")
        # Place it at the bottom row (F) of the grid
        self.place_in_area(problem_label, "F1", "F6", scale_factor=0.8)
        
        # Reveal the text
        self.play(Write(problem_label))
        self.wait(2)
