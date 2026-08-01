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
        # Teaching Content
        title_text = "Prerequisite 1: The Power of 'i' as a Rotation"
        lecture_lines = [
            'We start on the standard real number line.', 
            'Multiplying by i rotates us ninety degrees.', 
            'This rotation defines the vertical imaginary axis.'
        ]
        
        self.setup_layout(title_text, lecture_lines)
        
        # Colors
        REAL_COLOR = "#FFFFFF" # White
        IMAG_COLOR = "#FF69B4" # Pink
        TRANS_COLOR = "#FFFF00" # Yellow for the transition highlight
        
        # === Animation for Lecture Line 1 ===
        # We start on the standard real number line.
        self.lecture[0].set_color(REAL_COLOR)
        
        # Grid positions for the coordinate system
        origin_pos = self.grid["D3"]
        one_pos = self.grid["D4"]
        
        # Horizontal line (white)
        # Span from D1 to D6 on the 6x6 grid
        real_axis = Line(self.grid["D1"], self.grid["D6"], color=REAL_COLOR)
        
        # Small arrow pointing at the number 1
        arrow = Arrow(start=origin_pos, end=one_pos, buff=0, color=REAL_COLOR, stroke_width=4)
        
        # Label for the number 1
        label_1 = Text("1", font_size=24, color=REAL_COLOR)
        self.place_at_grid(label_1, "E5", scale_factor=1.0) 

        self.play(Create(real_axis))
        self.play(GrowArrow(arrow), Write(label_1))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Multiplying by i rotates us ninety degrees.
        self.lecture[1].set_color(TRANS_COLOR)
        
        # Rotate the arrow 90 degrees counter-clockwise around the origin
        # Note: D3 to D4 is unit distance horizontally; D3 to C3 is unit distance vertically
        self.play(
            Rotate(arrow, angle=PI/2, about_point=origin_pos),
            arrow.animate.set_color(TRANS_COLOR),
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # This rotation defines the vertical imaginary axis.
        self.lecture[2].set_color(IMAG_COLOR)
        
        # Pink vertical line intersecting the horizontal line
        # Vertical axis along column 3 (A3 to F3)
        imag_axis = Line(self.grid["A3"], self.grid["F3"], color=IMAG_COLOR)
        
        # Label for 'i'
        label_i = Text("i", font_size=24, color=IMAG_COLOR)
        self.place_at_grid(label_i, "C3", scale_factor=1.0) 
        
        self.play(
            Create(imag_axis), 
            Write(label_i),
            arrow.animate.set_color(IMAG_COLOR)
        )
        self.wait(2)
