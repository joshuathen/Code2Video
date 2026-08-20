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
        self.setup_layout("The Geometric Transition: Zooming In", [
            "Zoom into curves with a magnifying glass.",
            "Secant lines transform into local tangent slopes.",
            "The roller coaster segment becomes a straight line."
        ])

        # Assets
        curve = FunctionGraph(lambda x: 0.1 * x**3 - 0.2 * x**2 + 0.5, x_range=[-3, 3], color=WHITE)
        # Fix for issue 29/42: adjust curve placement
        self.place_in_area(curve, 'B4', 'E6', scale_factor=0.5)
        self.add(curve)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        # Fix for issue 22: use provided asset
        mag_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying.svg", color="#00FF00")
        # Fix for issue 30/42: adjust mag_glass placement
        self.place_at_grid(mag_glass, 'C5', scale_factor=0.7)
        self.play(Create(mag_glass))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        tangent = Line(start=np.array([-1, 0, 0]), end=np.array([1, 0, 0]), color="#FF00FF", stroke_width=6)
        # Fix for issue 31/42: adjust tangent placement
        self.place_at_grid(tangent, 'D5', scale_factor=0.7)
        self.play(Create(tangent))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        self.play(
            mag_glass.animate.scale(1.5),
            curve.animate.set_stroke(opacity=0.3),
            tangent.animate.scale(2.0).set_color(YELLOW)
        )
        self.wait(2)
