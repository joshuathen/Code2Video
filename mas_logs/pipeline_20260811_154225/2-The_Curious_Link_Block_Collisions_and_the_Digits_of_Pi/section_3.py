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

class Section3Scene(TeachingScene):
    def construct(self):
        self.setup_layout("The Geometry of Collisions: The Phase Space", [
            "Collision dynamics trace a circular arc.",
            "Each boundary hit reflects the movement path.",
            "Geometry encodes the system's ongoing state changes."
        ])
        
        # Assets
        ball = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ball.svg")
        
        # Define objects
        axes = Axes(x_range=[-2, 2, 1], y_range=[-2, 2, 1], axis_config={"include_tip": True})
        circle = Circle(radius=1.5, color=BLUE)
        path = Arc(radius=1.5, start_angle=PI/4, angle=PI/2, color="#FF00FF")
        boundary_line = Line(start=np.array([0, -2, 0]), end=np.array([0, 2, 0]), color=RED)
        
        # === Animation for Lecture Line 1 ===
        # Collision dynamics trace a circular arc.
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.place_in_area(axes, 'D1', 'F6', scale_factor=0.45)
        self.play(Create(axes))
        self.place_at_grid(ball, 'D3', scale_factor=0.2)
        self.add(ball)
        self.place_in_area(circle, 'D1', 'F6', scale_factor=0.5)
        self.play(Create(circle))
        self.play(Create(path))
        
        # === Animation for Lecture Line 2 ===
        # Each boundary hit reflects the movement path.
        self.play(self.lecture[1].animate.set_color("#FF00FF"))
        # Move ball to point on path
        self.play(ball.animate.move_to(path.point_from_proportion(0.5)))
        self.place_in_area(boundary_line, 'D1', 'F6', scale_factor=0.5)
        self.play(Create(boundary_line))

        # === Animation for Lecture Line 3 ===
        # Geometry encodes the system's ongoing state changes.
        self.play(self.lecture[2].animate.set_color(GREEN))
        self.play(Indicate(path), Indicate(circle))
