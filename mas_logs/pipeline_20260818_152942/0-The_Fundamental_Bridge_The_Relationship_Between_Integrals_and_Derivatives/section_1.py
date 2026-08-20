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
        self.setup_layout("Prerequisite Warm-up: The Concept of Change", [
            "Derivatives define instantaneous rates of change.",
            "Visualize the slope of a tangent line.",
            "Imagine a cheetah's speed at one moment."
        ])
        
        # Define objects
        curve = FunctionGraph(lambda x: 0.1 * x**3, x_range=[-3, 3])
        point = Dot(color=WHITE)
        point_label = MathTex("f(x)", font_size=24, color=WHITE)
        cheetah = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cheetah.svg")
        tangent = Line(start=[-1, -0.5, 0], end=[1, 0.5, 0], color="#FF0000")
        triangle = Polygon([-0.5, -0.25, 0], [0.5, -0.25, 0], [0.5, 0.25, 0], color="#00FF00", fill_opacity=0.3)
        slope_label = MathTex("dy/dx", font_size=24, color=WHITE)

        # Place objects
        self.place_in_area(curve, "A3", "C6", scale_factor=0.6)
        self.place_at_grid(point, "B3")
        self.place_at_grid(point_label, "A2", scale_factor=0.8)
        self.place_at_grid(cheetah, "B3", scale_factor=0.5)
        self.place_at_grid(tangent, "B3")
        self.place_at_grid(triangle, "D3")
        self.place_at_grid(slope_label, "D3", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFF00"), Write(curve), Write(point), Write(point_label), FadeIn(cheetah))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#FF0000"), Create(tangent))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#00FF00"), Create(triangle), Write(slope_label))
        
        self.wait(1)
        self.play(FadeOut(curve), FadeOut(point), FadeOut(point_label), FadeOut(cheetah), FadeOut(tangent), FadeOut(triangle), FadeOut(slope_label))
