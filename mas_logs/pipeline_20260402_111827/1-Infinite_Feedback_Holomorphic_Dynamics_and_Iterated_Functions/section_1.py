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
        # Title and Lecture Lines
        title = "The Complex Playground (Prerequisites)"
        lines = [
            "The complex plane provides our mathematical stage.",
            "Point z represents a coordinate in this space.",
            "A vector defines the position from the origin.",
            "Holomorphic functions transform the plane's geometry.",
            "Local angles remain preserved throughout the transformation."
        ]
        self.setup_layout(title, lines)

        # === Animation for Lecture Line 1 ===
        # Highlight: Matching colors (Grid #555555)
        self.lecture[0].set_color("#555555")
        
        # 1. Create a 2D coordinate grid in #555555 for the complex plane.
        plane = ComplexPlane(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            x_length=5,
            y_length=5,
            background_line_style={
                "stroke_color": "#555555",
                "stroke_width": 2,
                "stroke_opacity": 0.6
            }
        )
        # Issue 26 Fix: Avoid obstructing lecture notes
        self.place_in_area(plane, 'B1', 'F6', scale_factor=0.6)
        plane_origin = plane.get_origin()

        self.play(Create(plane))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight: Matching colors (Point/Label #FFFFFF)
        self.lecture[0].set_color(WHITE)
        self.lecture[1].set_color(WHITE)

        # 2. Fade in point 'z' at (2, 1) with label 'z = x + iy' in #FFFFFF
        z_val = 2 + 1j
        z_point = Dot(plane.n2p(z_val), color=WHITE)
        label_z = Text("z = x + iy", font_size=24, color=WHITE)
        # Issue 27 Fix: Keep label on-screen
        self.place_in_area(label_z, 'A5', 'B6', scale_factor=0.8)

        self.play(FadeIn(z_point), Write(label_z))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Highlight: Matching colors (Vector #00FFFF)
        self.lecture[1].set_color(WHITE)
        self.lecture[2].set_color("#00FFFF")

        # 3. Draw a #00FFFF vector from the origin to point 'z'
        z_vector = Arrow(plane_origin, z_point.get_center(), buff=0, color="#00FFFF", stroke_width=3)
        self.play(Create(z_vector))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        # Highlight: Transformation logic
        self.lecture[2].set_color(WHITE)
        self.lecture[3].set_color("#00FFFF")

        # 4. Transform the entire grid using f(z) = z^2 while preserving 90-degree intersections.
        def complex_func(z):
            return z**2

        def plane_transform(p):
            z_c = plane.p2n(p)
            w_c = complex_func(z_c)
            return plane.n2p(w_c)

        new_z_val = complex_func(z_val)
        new_z_pos = plane.n2p(new_z_val)

        self.play(
            plane.animate.apply_function(plane_transform),
            z_point.animate.move_to(new_z_pos),
            z_vector.animate.put_start_and_end_on(plane_origin, new_z_pos),
            run_time=3
        )
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        # Highlight: Angle preservation color #FFFF00
        self.lecture[3].set_color(WHITE)
        self.lecture[4].set_color("#FFFF00")

        # 5. Zoom in on a grid intersection to highlight angle preservation in #FFFF00.
        angle_indicator = VGroup(
            Line(ORIGIN, 0.5 * RIGHT),
            Line(ORIGIN, 0.5 * UP)
        ).set_color("#FFFF00")
        
        # Issue 28 Fix: Avoid overlapping lecture text
        self.place_in_area(angle_indicator, 'C3', 'E5', scale_factor=0.7)
        angle_indicator.rotate(30 * DEGREES) # Aesthetic rotation for transformed space

        zoom_center = angle_indicator.get_center()
        # Simulation of zoom by scaling relative to the indicator
        everything_else = VGroup(plane, z_point, z_vector, label_z)

        self.play(Create(angle_indicator))
        self.play(
            everything_else.animate.scale(2.5, about_point=zoom_center),
            angle_indicator.animate.scale(1.5),
            run_time=2
        )
        self.wait(2)
