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

class Section5Scene(TeachingScene):
    def construct(self):
        # Initial layout setup
        self.setup_layout(
            "The Journey of π Radians", 
            [
                "In Euler's formula, the exponent is the distance traveled.", 
                "A full trip around is two pi radians.", 
                "Traveling exactly pi takes us halfway around."
            ]
        )

        # Define colors as specified in the prompt
        CIRCLE_COLOR = "#AAAAAA"
        ARC_COLOR = "#FF5555"

        # Preparation of complex plane and circle
        plane = ComplexPlane(
            x_range=[-2, 2, 1], 
            y_range=[-2, 2, 1],
            x_length=3,
            y_length=3,
            background_line_style={"stroke_opacity": 0.4}
        )
        
        # Calculate unit size for radius consistency
        unit_size = plane.get_x_unit_size()
        
        circle = Circle(radius=unit_size, color=CIRCLE_COLOR, stroke_opacity=0.6)
        # Position dot at (1,0) on the unit circle
        dot_start = Dot(plane.n2p(1), color=CIRCLE_COLOR)
        
        # Prepare Arc and π label
        # Arc from 0 to PI (180 degrees)
        arc = Arc(
            radius=unit_size, 
            start_angle=0, 
            angle=PI, 
            color=ARC_COLOR, 
            stroke_width=6
        )
        # pi label
        label_pi = Text("π", color=ARC_COLOR)

        # Group components to ensure they move together and maintain relative positions
        vis_group = VGroup(plane, circle, dot_start, arc)
        
        # Issue 48 Fix: Move vis_group further to the right to prevent crowding
        self.place_in_area(vis_group, 'B3', 'E6')
        
        # Issue 49 Fix: Scale pi label down and position correctly
        self.place_at_grid(label_pi, 'C2', scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        # Line 1 color change and initial visuals
        self.play(self.lecture[0].animate.set_color(CIRCLE_COLOR))
        self.play(
            Create(plane), 
            Create(circle), 
            FadeIn(dot_start), 
            run_time=1.5
        )
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Line 2 color change
        self.play(self.lecture[1].animate.set_color(CIRCLE_COLOR))
        # Narrative focus on the concept of 2pi for a full circle
        self.wait(2)

        # === Animation for Lecture Line 3 ===
        # Line 3 color change to match arc
        self.play(self.lecture[2].animate.set_color(ARC_COLOR))
        
        # Trace the red arc counter-clockwise from (1,0) to (-1,0)
        self.play(Create(arc), run_time=2.5, rate_func=linear)
        
        # Appearance of the pi label
        self.play(Write(label_pi))
        self.wait(3)
