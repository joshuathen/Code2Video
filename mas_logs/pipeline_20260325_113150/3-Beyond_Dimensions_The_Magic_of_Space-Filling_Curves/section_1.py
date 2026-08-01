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
        # 1. Setup layout with section title and lecture lines
        lecture_lines = [
            "Can a one-dimensional line fill a two-dimensional square?",
            "Imagine an ant painting with infinitely thin thread.",
            "It must touch every point without leaving gaps."
        ]
        self.setup_layout("The Infinite Ant: A Geometric Paradox", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Draw a white square outline (#FFFFFF) at the center. 
        # Place a small orange dot (#FFA500) representing the ant at the bottom-left corner.
        
        # Defining a square that fits within the B2 to E5 grid area (approx 3x3 units)
        square = Square(side_length=3.0, color=WHITE)
        self.place_in_area(square, "B2", "E5")
        
        # The ant starts at the bottom-left corner of the square (corresponds to grid E2)
        ant = Dot(color="#FFA500", radius=0.1)
        self.place_at_grid(ant, "E2")
        
        # Ensure the first lecture line starts as white (default)
        self.play(self.lecture[0].animate.set_color(WHITE))
        self.play(Create(square))
        self.play(FadeIn(ant))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # The orange dot moves rapidly in a complex, erratic path inside the square, 
        # leaving behind a thin white line (#FFFFFF).
        
        # Define an erratic path using the square's geometry
        center = square.get_center()
        points = [
            square.get_corner(DL),
            center + UP * 0.7 + LEFT * 1.1,
            center + DOWN * 0.4 + LEFT * 0.2,
            center + UP * 1.0 + RIGHT * 0.3,
            center + DOWN * 1.1 + RIGHT * 0.8,
            center + UP * 0.1 + RIGHT * 1.2,
            square.get_corner(DR) + UP * 0.4,
            square.get_corner(UR) + LEFT * 0.4,
            center + LEFT * 0.6 + DOWN * 0.2,
        ]
        
        path = VMobject(color=WHITE, stroke_width=2)
        path.set_points_as_corners(points)
        
        # Highlight second lecture line with the color of the "ant" (Orange)
        self.play(self.lecture[1].animate.set_color("#FFA500"))
        
        # Trace the path with the ant
        self.play(
            MoveAlongPath(ant, path),
            Create(path),
            run_time=4,
            rate_func=linear
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # The movement stops. A red circle (#FF0000) appears around a black gap (#000000) 
        # in the square, with the label 'Gap' (#FF0000) nearby.
        
        # Choose a spot within the square for the gap indicator
        # Issue 27: Scale gap_circle by 0.7 at D4
        gap_circle = Circle(radius=0.3, color="#FF0000")
        self.place_at_grid(gap_circle, "D4", scale_factor=0.7)
        
        # Issue 26: Move label to D5 and scale by 0.6
        gap_label = Text("Gap", font_size=24, color="#FF0000")
        self.place_at_grid(gap_label, "D5", scale_factor=0.6)
        
        # Highlight third lecture line with the color of the gap indicators (Red)
        self.play(self.lecture[2].animate.set_color("#FF0000"))
        
        # Show gap highlight and label
        self.play(Create(gap_circle))
        self.play(Write(gap_label))
        self.wait(2)
