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
        # Define high-level context
        title = "Prerequisite: The Complex Playground"
        lecture_lines = [
            'The complex plane has real and imaginary axes.', 
            'Multiplying by i rotates us 90 degrees counter-clockwise.', 
            'It is a playground where numbers can turn and spin.'
        ]
        self.setup_layout(title, lecture_lines)

        # Highlight Colors
        H_COLOR_1 = "#FFFFFF" # White
        H_COLOR_2 = "#00FFFF" # Cyan for rotation
        H_COLOR_3 = "#FFD700" # Gold for spinning playground

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(H_COLOR_1)
        
        # Origin is at D4
        origin = self.grid["D4"]
        
        # Real Axis (Horizontal) from D2 to D6
        real_axis = Line(start=self.grid["D2"], end=self.grid["D6"], color=WHITE)
        # Imaginary Axis (Vertical) from F4 to A4
        imag_axis = Line(start=self.grid["F4"], end=self.grid["A4"], color=WHITE)
        
        # Labels for axes
        re_label = Text("Real", color=WHITE, font_size=24)
        self.place_at_grid(re_label, "E6", scale_factor=0.8) # Fix Issue 27
        
        im_label = Text("Imaginary", color=WHITE, font_size=24)
        self.place_at_grid(im_label, "A5", scale_factor=0.8) # Fix Issue 28

        self.play(
            Create(real_axis), 
            Create(imag_axis), 
            Write(re_label), 
            Write(im_label),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(H_COLOR_2)
        
        # Dot at (1, 0) corresponds to grid D5
        dot = Dot(point=self.grid["D5"], color=WHITE)
        one_label = Text("1", color=WHITE, font_size=24)
        self.place_at_grid(one_label, "E5", scale_factor=0.8) # Fix Issue 29
        
        i_label = Text("i", color=H_COLOR_2, font_size=24, slant=ITALIC)
        self.place_at_grid(i_label, "B4", scale_factor=0.8)

        # Rotation Arc from D5 to C4 around D4
        rotation_arc = Arc(
            radius=1.0, 
            start_angle=0, 
            angle=PI/2, 
            arc_center=origin, 
            color=H_COLOR_2
        )

        self.play(FadeIn(dot), Write(one_label))
        self.wait(0.5)
        
        self.play(
            MoveAlongPath(dot, rotation_arc),
            Create(rotation_arc),
            run_time=2
        )
        self.play(Write(i_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color(H_COLOR_3)
        
        # Visualize the "playground" with further rotations
        # -1 at D3, -i at E4
        minus_one_label = Text("-1", color=H_COLOR_3, font_size=24)
        self.place_at_grid(minus_one_label, "E3", scale_factor=0.8)
        
        minus_i_label = Text("-i", color=H_COLOR_3, font_size=24, slant=ITALIC)
        self.place_at_grid(minus_i_label, "F4", scale_factor=0.8)

        arc_to_neg1 = Arc(
            radius=1.0, 
            start_angle=PI/2, 
            angle=PI/2, 
            arc_center=origin, 
            color=H_COLOR_3
        )
        arc_to_negi = Arc(
            radius=1.0, 
            start_angle=PI, 
            angle=PI/2, 
            arc_center=origin, 
            color=H_COLOR_3
        )
        
        # Spinning silver arrow
        spinning_arrow = Arc(
            radius=1.5, 
            start_angle=0, 
            angle=1.5*PI, 
            arc_center=origin, 
            color="#C0C0C0"
        ).add_tip()
        
        self.play(
            MoveAlongPath(dot, arc_to_neg1),
            Create(arc_to_neg1),
            Write(minus_one_label),
            run_time=1.5
        )
        
        self.play(
            MoveAlongPath(dot, arc_to_negi),
            Create(arc_to_negi),
            Write(minus_i_label),
            run_time=1.5
        )

        self.play(
            Create(spinning_arrow),
            Rotate(spinning_arrow, angle=TAU, about_point=origin, rate_func=linear),
            run_time=2
        )

        self.wait(2)
