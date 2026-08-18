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
            "Column space is the reach of transformation.",
            "It spans all possible reachable destinations.",
            "Visualize this as a reachable region."
        ]
        self.setup_layout("Column Space: The Reach of a Transformation", lecture_lines)
        
        # Setup 3D axes
        axes = ThreeDAxes(x_range=[-3, 3], y_range=[-3, 3], z_range=[-3, 3], axis_config={"include_tip": True})
        self.place_in_area(axes, 'C2', 'F6', scale_factor=0.5)
        
        # Assets
        globe = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/globe.svg")
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")
        
        # === Animation for Lecture Line 1 ===
        # Display column vectors in 3D
        vec1 = Vector([2, 1, 0], color="#FF00FF")
        vec2 = Vector([0, 1, 2], color="#FFFF00")
        
        self.place_at_grid(globe, "B3", scale_factor=0.5)
        
        self.play(self.lecture[0].animate.set_color("#FF00FF"))
        self.add(axes)
        self.play(Create(vec1), Create(vec2), FadeIn(globe))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Span the space covered
        plane = Polygon(
            axes.c2p(0, 0, 0), 
            axes.c2p(2, 1, 0), 
            axes.c2p(2, 2, 2), 
            axes.c2p(0, 1, 2), 
            fill_opacity=0.3, color="#666666"
        )
        self.play(self.lecture[1].animate.set_color("#666666"))
        self.play(Create(plane))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Indicate output vectors lie within this span
        dot = Dot(axes.c2p(1, 1.5, 1), color=RED)
        label = Text("Reachable", font_size=16, color=RED)
        self.place_at_grid(label, 'D5', scale_factor=0.7)
        self.place_at_grid(map_icon, "C6", scale_factor=0.4)
        
        self.play(self.lecture[2].animate.set_color(RED))
        self.play(FadeIn(dot), Write(label), FadeIn(map_icon))
        self.wait(2)
