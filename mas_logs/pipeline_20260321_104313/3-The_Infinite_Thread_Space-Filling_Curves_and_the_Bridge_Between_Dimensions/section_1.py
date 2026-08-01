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
        # Initialize layout
        lecture_lines = [
            "In geometry, we distinguish between 1D lines and 2D planes.",
            "A line has length but occupies no area.",
            "Can a 1D thread ever completely fill a 2D square?"
        ]
        self.setup_layout("The Dimensional Paradox", lecture_lines)

        # === Animation for Lecture Line 1 ===
        # Highlight first lecture line
        self.play(self.lecture[0].animate.set_color("#00FFFF"))
        
        # Create 1D representation (Line)
        # Using a vertical line on the 'left' side of the right-hand grid
        line_1d = Line(UP, DOWN, color=WHITE)
        self.place_in_area(line_1d, "A2", "C2")
        label_1d = Text("1D", font_size=24, color=WHITE)
        self.place_at_grid(label_1d, "D2")
        
        # Create 2D representation (Square)
        # Using a square on the 'right' side of the right-hand grid
        square_2d = Square(side_length=2.0, color=WHITE)
        self.place_in_area(square_2d, "A5", "C5")
        label_2d = Text("2D", font_size=24, color=WHITE)
        self.place_at_grid(label_2d, "D5")
        
        self.play(Create(line_1d), Write(label_1d))
        self.play(Create(square_2d), Write(label_2d))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Revert color 1, highlight color 2
        self.play(
            self.lecture[0].animate.set_color(WHITE),
            self.lecture[1].animate.set_color("#00FFFF")
        )
        
        # Pulse animation for the 1D line
        self.play(line_1d.animate.set_color("#00FFFF"))
        self.play(line_1d.animate.scale(1.2), run_time=0.4)
        self.play(line_1d.animate.scale(1/1.2), run_time=0.4)
        
        # Show grid inside the square to emphasize the 2D area
        grid_in_square = NumberPlane(
            x_range=[0, 4, 1], y_range=[0, 4, 1],
            x_length=2.0, y_length=2.0,
            background_line_style={"stroke_color": GRAY, "stroke_width": 1}
        )
        self.place_in_area(grid_in_square, "A5", "C5")
        self.play(Create(grid_in_square))

        # Move/Transform the 1D line to sit inside the square area
        # We transform it into a horizontal line to show it spans length but no area relative to the square
        target_horizontal = Line(LEFT, RIGHT, color="#00FFFF")
        self.place_in_area(target_horizontal, "A5", "C5") # Centered inside square
        
        self.play(
            Transform(line_1d, target_horizontal),
            FadeOut(label_1d)
        )
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Revert color 2, highlight color 3
        self.play(
            self.lecture[1].animate.set_color(WHITE),
            self.lecture[2].animate.set_color("#00FFFF")
        )
        
        # Helper to generate a folding/zigzag pattern within local bounds [-1, 1]
        def get_zigzag_mobject(n):
            vertices = []
            # Local bounds to fit square side_length=2
            step = 2.0 / n
            for i in range(n + 1):
                y = 1.0 - i * step
                if i % 2 == 0:
                    vertices.append([-1, y, 0])
                    vertices.append([1, y, 0])
                else:
                    vertices.append([1, y, 0])
                    vertices.append([-1, y, 0])
            return VMobject().set_points_as_corners(vertices).set_color("#00FFFF")

        # Create simple and then complex foldings
        zigzag_1 = get_zigzag_mobject(4)
        self.place_in_area(zigzag_1, "A5", "C5")
        
        zigzag_2 = get_zigzag_mobject(20)
        self.place_in_area(zigzag_2, "A5", "C5")

        # Execute folding animation
        self.play(Transform(line_1d, zigzag_1))
        self.play(Transform(line_1d, zigzag_2), run_time=2.5)
        self.wait(2)
