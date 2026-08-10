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
        lecture_lines = ["A linear combination scales and adds vectors.", 
                         "Span is every point a combination can reach.", 
                         "Two non-parallel vectors sweep the entire plane."]
        self.setup_layout("Linear Combinations and Span", lecture_lines)
        
        # Setup vectors and container
        u = Vector([1, 0.5], color=BLUE)
        v = Vector([-0.5, 1], color=YELLOW)
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": False}).scale(0.5)
        span_area = Polygon(
            axes.c2p(-3, -3), axes.c2p(3, -3), axes.c2p(3, 3), axes.c2p(-3, 3),
            fill_color="#808080", fill_opacity=0.3, stroke_width=0
        )
        
        # Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg
        # Loading as SVGMobject
        plane_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/plane.svg")
        
        grid_group = VGroup(axes, u, v, span_area, plane_icon)
        self.place_in_area(grid_group, 'B4', 'E6', scale_factor=0.5)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(Create(axes), GrowArrow(u), GrowArrow(v))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        dot = Dot(color=RED)
        self.add(dot)
        self.play(dot.animate.move_to(axes.c2p(1, 1)))
        self.play(dot.animate.move_to(axes.c2p(-0.5, 0.5)))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(GRAY)
        self.play(FadeIn(span_area), FadeIn(plane_icon))
        self.wait(2)
