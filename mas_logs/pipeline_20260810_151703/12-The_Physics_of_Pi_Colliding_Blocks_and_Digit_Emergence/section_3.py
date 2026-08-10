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
        lecture_lines = [
            "Plot velocities as two-dimensional coordinates.",
            "Collisions appear as boundary reflections.",
            "Geometry solves the physical system."
        ]
        self.setup_layout("Mapping Collisions to Phase Space", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        # Draw a 2D plane with X and Y axes, color #808080.
        axes = Axes(
            x_range=[0, 5, 1],
            y_range=[0, 5, 1],
            axis_config={"color": "#808080"}
        )
        self.place_at_grid(axes, "C3", scale_factor=0.6)
        self.play(Create(axes))
        self.lecture[0].set_color("#808080")

        # === Animation for Lecture Line 2 ===
        # Represent collision as a point hitting a boundary line [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/boundary.svg], color #00FF00.
        boundary = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/boundary.svg")
        boundary.set_color("#00FF00")
        self.place_at_grid(boundary, "C4", scale_factor=0.5)
        
        dot = Dot(color="#00FF00")
        dot.move_to(axes.c2p(0.5, 0.5))
        
        self.play(FadeIn(boundary), FadeIn(dot))
        self.lecture[1].set_color("#00FF00")

        # === Animation for Lecture Line 3 ===
        # Show the geometric path reflecting in the phase space, color #FFFFFF.
        path = Line(axes.c2p(0.5, 0.5), axes.c2p(2, 2), color="#FFFFFF")
        reflection = Line(axes.c2p(2, 2), axes.c2p(3.5, 0.5), color="#FFFFFF")
        
        self.play(Create(path), MoveAlongPath(dot, path))
        self.play(Create(reflection), MoveAlongPath(dot, reflection))
        self.lecture[2].set_color("#FFFFFF")
        self.wait(2)
