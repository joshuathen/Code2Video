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
        # Setup the layout with title and lecture lines
        self.setup_layout(
            "Prerequisite: The Magic of 'i' as a Turn", 
            [
                "Forget 'impossible'; think of i as a spatial turn.",
                "Multiplying by i rotates a number ninety degrees counter-clockwise.",
                "This shifts our view from lines to the complex plane."
            ]
        )

        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1 in green to match the dot
        self.lecture[0].set_color("#00FF00")
        
        # Draw a white horizontal number line. 
        # Using grid points D1 to D6 to span the right-side area.
        h_axis = Line(self.grid["D1"], self.grid["D6"], color=WHITE)
        
        # Place a green dot at the position '1' (grid D5, with origin at D3)
        origin_point = self.grid["D3"]
        pos_1 = self.grid["D5"]
        dot = Dot(pos_1, color="#00FF00")
        
        self.play(Create(h_axis))
        self.play(FadeIn(dot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Reset previous line and highlight current line in magenta
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color("#FF00FF")
        
        # Define the rotation arc (radius 2 grid units, from D5 to B3 around D3)
        arc = Arc(
            radius=2.0, 
            start_angle=0, 
            angle=PI/2, 
            arc_center=origin_point, 
            color="#FF00FF"
        )
        
        # Animate the dot rotating 90 degrees CCW
        self.play(
            Create(arc), 
            Rotate(dot, angle=PI/2, about_point=origin_point), 
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Reset previous line and highlight current line in magenta
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#FF00FF")
        
        # Draw the vertical axis through the origin (A3 to F3)
        v_axis = Line(self.grid["A3"], self.grid["F3"], color=WHITE)
        
        # Label the new vertical position as 'i'
        label_i = Text("i", font_size=32, color="#FF00FF")
        # Fix Issue 32: Scale reduced to 0.6
        self.place_at_grid(label_i, "B2", scale_factor=0.6)
        
        # Show text describing the operation
        op_text = Text("x i = +90 rotation", font_size=24, color="#FF00FF")
        # Fix Issue 31: use place_in_area to avoid clipping
        self.place_in_area(op_text, "A4", "A6", scale_factor=0.7)
        
        self.play(
            Create(v_axis),
            Write(label_i),
            Write(op_text)
        )
        self.wait(2)
