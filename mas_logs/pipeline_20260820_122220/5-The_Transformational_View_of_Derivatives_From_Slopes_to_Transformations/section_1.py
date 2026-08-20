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
        self.setup_layout("Prerequisite Review: The Static Slope", [
            "Derivative is the slope of the tangent line.",
            "Imagine a static curve with a fixed ruler.",
            "The slope represents the rate at a point."
        ])
        
        # Define objects
        axes = Axes(x_range=[0, 3, 1], y_range=[0, 3, 1], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.5 * x**2, color=WHITE)
        point = axes.c2p(2, 2)
        slope = 1.0 * 2 # f'(2) = x evaluated at 2
        
        tangent = Line(start=axes.c2p(1, 1), end=axes.c2p(3, 3), color=YELLOW)
        ruler = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/ruler.svg")
        
        # Slope triangle
        p1 = axes.c2p(1.5, 1.125)
        p2 = axes.c2p(2.5, 2.125)
        tri = Polygon(
            axes.c2p(1.5, 1.125),
            axes.c2p(2.5, 1.125),
            axes.c2p(2.5, 2.125),
            color="#00FFFF"
        )
        
        self.place_in_area(axes, "B1", "E5", scale_factor=0.6)
        self.place_at_grid(curve, "C3", scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(curve), run_time=1)
        self.play(Create(tangent), run_time=1)
        self.place_at_grid(ruler, "D4", scale_factor=0.5)
        self.play(FadeIn(ruler))
        self.lecture[0].set_color(YELLOW)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(Create(tri))
        self.lecture[2].set_color("#00FFFF")
        self.play(Flash(tri, color="#00FFFF", line_stroke_width=2))
        self.wait(1)
