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

class Section3Scene(TeachingScene):
    def construct(self):
        # Fetching data from shared state (handled in prompt, but variables defined here)
        title = "Component Form: The DNA of a Vector"
        lines = [
            "Every vector breaks down into horizontal and vertical parts.",
            "First, we identify the horizontal move along the x-axis.",
            "Next, we measure the vertical move along the y-axis.",
            "We write these components inside square brackets like this.",
            "These parts form a right triangle with the vector."
        ]
        self.setup_layout(title, lines)

        # Pre-define colors for highlights
        COLOR_V = "#00FFFF"  # Cyan
        COLOR_X = "#FF8800"  # Orange
        COLOR_Y = "#00FF00"  # Green
        COLOR_RA = "#FFFFFF" # White

        # Define key points in the grid
        # Vector from (0,0) to (4,3)
        # Using E2 as (0,0)
        # E2 to E6 is (4,0) - length 4
        # E6 to B6 is (0,3) - length 3
        origin = self.grid["E2"]
        x_end = self.grid["E6"]
        tip = self.grid["B6"]

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(COLOR_V)
        
        vector_v = Arrow(start=origin, end=tip, buff=0, color=COLOR_V, stroke_width=4)
        label_v = MathTex(r"\vec{v}", color=COLOR_V)
        # Positioned near the tip
        self.place_at_grid(label_v, "B5", scale_factor=0.8)
        
        self.play(Create(vector_v), Write(label_v))
        self.wait(2.0)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(COLOR_X)
        
        line_x = Line(start=origin, end=x_end, color=COLOR_X, stroke_width=5)
        label_x = MathTex("x=4", color=COLOR_X)
        # FIX Issue 34: Moved to F4 to avoid overlap with line along row E
        self.place_at_grid(label_x, "F4", scale_factor=0.8)
        
        self.play(Create(line_x), Write(label_x))
        self.wait(1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(COLOR_Y)
        
        line_y = Line(start=x_end, end=tip, color=COLOR_Y, stroke_width=5)
        label_y = MathTex("y=3", color=COLOR_Y)
        # FIX Issue 35: Moved to D5 to avoid overlap with line along column 6
        self.place_at_grid(label_y, "D5", scale_factor=0.8)
        
        self.play(Create(line_y), Write(label_y))
        self.wait(1.5)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(WHITE)
        
        bracket_form = MathTex(r"\vec{v} = \begin{bmatrix} 4 \\ 3 \end{bmatrix}", color=WHITE)
        # FIX Issue 33: Moved to area A3-B4 to avoid overlap with lecture line text
        self.place_in_area(bracket_form, "A3", "B4", scale_factor=0.9)
        
        # Morphing labels into bracket notation
        label_x_copy = label_x.copy()
        label_y_copy = label_y.copy()
        
        self.play(
            ReplacementTransform(label_x_copy, bracket_form),
            ReplacementTransform(label_y_copy, bracket_form),
            run_time=2
        )
        self.wait(2.0)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(COLOR_RA)
        
        # Right angle symbol at (4,0) which is grid E6
        # quadrant=(-1, 1) ensures it's inside the triangle (left and up from E6)
        right_angle = RightAngle(line_x, line_y, length=0.3, quadrant=(-1, 1), color=COLOR_RA)
        
        self.play(Create(right_angle))
        self.wait(2.0)
