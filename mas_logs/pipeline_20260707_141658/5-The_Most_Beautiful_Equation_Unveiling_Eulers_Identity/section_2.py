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
        # Title and Lecture Lines
        title_text = "Prerequisite 1: The Secret Life of 'i'"
        lecture_lines = [
            "The unit i is defined as square root of -1.",
            "Geometrically, multiplying by i represents a 90-degree rotation.",
            "It turns East into North on the complex plane."
        ]
        self.setup_layout(title_text, lecture_lines)
        
        # Colors (Light, distinguishable hexadecimal)
        color1 = "#FFFFFF" # White
        color2 = "#FFFFE0" # Yellow
        color3 = "#ADD8E6" # Light Blue
        
        # === Animation for Lecture Line 1 ===
        # Highlight lecture line 1
        self.play(self.lecture[0].animate.set_color(color1))
        
        # Create complex plane
        plane = ComplexPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            axis_config={"include_tip": True, "stroke_width": 2},
            background_line_style={"stroke_opacity": 0.3}
        )
        # Fix for Issue 27: Position plane in area A2 to F6
        self.place_in_area(plane, 'A2', 'F6', scale_factor=0.8)
        
        # Initial vector at (1, 0)
        vector = Arrow(plane.n2p(0), plane.n2p(1), buff=0, color=color1, stroke_width=4)
        # Using Text to avoid LaTeX issues
        label_1 = Text("1", color=color1).scale(0.8)
        label_1.next_to(plane.n2p(1), DR, buff=0.1)
        
        # Asset: Compass integration for Issue 22
        compass = SVGMobject("/mmfs1/data/group/pmc082/jthen/Code2Video/assets/icon/compass.svg")
        self.place_at_grid(compass, 'A6', scale_factor=0.6)
        
        self.play(Create(plane), GrowArrow(vector), FadeIn(label_1), FadeIn(compass))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight lecture line 2
        self.play(self.lecture[1].animate.set_color(color2))
        
        # Label for 'i' at (0, 1)
        label_i = Text("i", color=color2)
        # Fix for Issue 28: Positioning label_i
        self.place_at_grid(label_i, 'B4', scale_factor=0.7)
        
        # Rotate vector to (0, 1) and compass to North
        self.play(
            Rotate(vector, angle=PI/2, about_point=plane.n2p(0)),
            Rotate(compass, angle=PI/2),
            vector.animate.set_color(color2),
            FadeIn(label_i),
            run_time=2
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight lecture line 3
        self.play(self.lecture[2].animate.set_color(color3))
        
        # Label for '-1' at (-1, 0)
        label_neg_1 = Text("-1", color=color3)
        # Fix for Issue 29: Positioning label_neg_1
        self.place_at_grid(label_neg_1, 'D3', scale_factor=0.7)
        
        # Rotate vector further to (-1, 0) and compass to West
        self.play(
            Rotate(vector, angle=PI/2, about_point=plane.n2p(0)),
            Rotate(compass, angle=PI/2),
            vector.animate.set_color(color3),
            FadeIn(label_neg_1),
            run_time=2
        )
        self.wait(2)
