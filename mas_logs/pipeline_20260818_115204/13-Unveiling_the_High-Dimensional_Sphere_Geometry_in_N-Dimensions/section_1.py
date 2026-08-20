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
        self.setup_layout("The Dimensional Ladder", [
            "A 2D circle is defined by points equidistant from center.",
            "Distance formula generalizes this to 3D spheres.",
            "Equation x² + y² + z² = r² defines a 3D ball."
        ])
        
        # Elements
        point = Dot(color=WHITE)
        line = Line(LEFT*0.5, RIGHT*0.5, color=WHITE)
        circle = Circle(radius=0.5, color=WHITE)
        # Using SVG asset
        sphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color=WHITE)
        
        labels = VGroup(
            Text("0D", font_size=18),
            Text("1D", font_size=18),
            Text("2D", font_size=18),
            Text("3D", font_size=18)
        )

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(point), Write(self.place_at_grid(labels[0], 'B1', 0.8)))
        self.play(Create(line), Write(self.place_at_grid(labels[1], 'B3', 0.8)))
        self.play(Create(circle), Write(self.place_at_grid(labels[2], 'B5', 0.8)))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        # Rotating sphere asset
        self.play(FadeIn(self.place_at_grid(sphere, 'D3', scale_factor=0.5)), Write(self.place_at_grid(labels[3], 'D3', scale_factor=0.8)))
        self.play(Rotate(sphere, angle=PI))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        equation = MathTex("x^2 + y^2 + z^2 = r^2", color="#FFD700")
        self.place_in_area(equation, 'D3', 'F5', scale_factor=0.6)
        
        # Final flash
        self.play(Write(equation), Flash(sphere, color="#FFD700"))
        
        self.wait(2)
