from manim import *
import numpy as np

class Section5Scene(Scene):
    def setup_layout(self, title_text, lecture_lines):
        # Background and Title
        self.camera.background_color = "#000000"
        self.title = Text(title_text, font_size=28, color=WHITE).to_edge(UP, buff=0.5)
        
        # Left-side lecture content
        lecture_texts = [Text(line, font_size=22, color=WHITE) for line in lecture_lines]
        self.lecture = VGroup(*lecture_texts).arrange(DOWN, aligned_edge=LEFT, buff=0.4)
        self.lecture.scale(0.85).to_edge(LEFT, buff=0.5)
        
        # Fine-grained animation grid (6x6 grid on right side)
        self.grid = {}
        rows = ["A", "B", "C", "D", "E", "F"]  # Top to bottom
        cols = ["1", "2", "3", "4", "5", "6"]  # Left to right

        for i, row in enumerate(rows):
            for j, col in enumerate(cols):
                # Mapping grid to the right side of the screen
                x = 1.0 + j * 0.9
                y = 2.2 - i * 0.9
                self.grid[f"{row}{col}"] = np.array([x, y, 0])

    def construct(self):
        # Configuration and initialization
        title_string = "Group Theory: The Visual Harmony of Euler's Identity"
        lecture_items = [
            "- Rotation as a Group Operation",
            "- The Unit Circle in the Complex Plane",
            "- Mapping SO(2) to Complex Numbers",
            "- Geometric Interpretation of e^iπ"
        ]
        
        self.setup_layout(title_string, lecture_items)
        
        # Add Title and Lecture notes
        self.add(self.title)
        self.play(FadeIn(self.lecture, shift=RIGHT))
        
        # Mathematical Visualization on the Grid
        # Use grid points C3 as the center for a complex plane representation
        plane_center = self.grid["C3"]
        plane = ComplexPlane(
            x_range=[-2, 2, 1],
            y_range=[-2, 2, 1],
            x_length=4,
            y_length=4,
            background_line_style={"stroke_opacity": 0.4}
        ).move_to(plane_center + RIGHT * 0.5)
        
        circle = Circle(radius=plane.get_x_unit_size(), color=BLUE_B)
        circle.move_to(plane.get_center())
        
        # Vector representing the rotation
        start_point = plane.n2p(1 + 0j)
        end_point = plane.n2p(-1 + 0j)
        
        pointer = Dot(start_point, color=YELLOW)
        label_start = Text("1", font_size=24).next_to(start_point, UR, buff=0.1)
        label_end = Text("-1", font_size=24).next_to(end_point, UL, buff=0.1)
        
        # Path for rotation (e^it)
        arc_path = Arc(
            radius=plane.get_x_unit_size(),
            start_angle=0,
            angle=PI,
            color=YELLOW,
            arc_center=plane.get_center()
        )
        
        # Animation sequence
        self.play(Create(plane), Create(circle))
        self.play(FadeIn(pointer), Write(label_start))
        self.wait(1)
        
        # Show the rotation from 1 to -1
        self.play(
            MoveAlongPath(pointer, arc_path),
            Create(arc_path),
            run_time=2.5,
            rate_func=smooth
        )
        self.play(Write(label_end))
        
        # Final formula display
        # Use Text instead of MathTex to avoid LaTeX dependency error
        formula = Text("e^iπ = -1", color=YELLOW, font_size=42)
        formula.move_to(self.grid["F3"] + RIGHT * 0.5)
        self.play(Write(formula))
        
        self.wait(3)