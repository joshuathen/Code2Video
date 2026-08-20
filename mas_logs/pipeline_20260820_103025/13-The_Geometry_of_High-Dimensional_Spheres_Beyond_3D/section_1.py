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
        lecture_lines = [
            "Distance in 2D is the Pythagorean hypotenuse.",
            "In 3D, we add a depth squared term.",
            "For n-dimensions, just sum all squared coordinates.",
            "A sphere holds points at a fixed distance.",
            "Equation: the sum of x_i squared equals radius squared."
        ]
        self.setup_layout("Prerequisites: The N-Dimensional Coordinate System", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Distance in 2D is the Pythagorean hypotenuse.
        self.lecture[0].set_color("#FFFFFF")
        axes = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2], axis_config={"include_tip": False})
        self.place_in_area(axes, 'A1', 'C3', scale_factor=0.4)
        self.play(Create(axes), run_time=1)

        # === Animation for Lecture Line 2 ===
        # In 3D, we add a depth squared term.
        self.lecture[1].set_color("#FFFFFF")
        # Visualizing a point in 3D
        dot = Dot(color="#FF00FF")
        self.place_at_grid(dot, "D3", scale_factor=1.0)
        self.play(FadeIn(dot))

        # === Animation for Lecture Line 3 ===
        # For n-dimensions, just sum all squared coordinates.
        self.lecture[2].set_color("#00FFFF")
        label = Text("n-dimensional representation", font_size=24, color="#00FFFF")
        self.place_in_area(label, 'E3', 'F5', scale_factor=0.7)
        self.play(Write(label))

        # === Animation for Lecture Line 4 ===
        # A sphere holds points at a fixed distance.
        self.lecture[3].set_color("#FF4500")
        sphere_asset = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#FF4500")
        self.place_at_grid(sphere_asset, "D4", scale_factor=0.8)
        self.play(FadeIn(sphere_asset))

        # === Animation for Lecture Line 5 ===
        # Equation: the sum of x_i squared equals radius squared.
        self.lecture[4].set_color("#FFD700")
        eq = MathTex(r"\sum x_i^2 = r^2", font_size=32, color="#FFD700")
        self.place_in_area(eq, 'A4', 'B6', scale_factor=0.8)
        self.play(FadeIn(eq))
        
        self.wait(2)
