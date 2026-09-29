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
            "Zoom in to see a curve's slope.",
            "A tiny section looks like a straight line.",
            "This slope is the derivative at that point.",
            "The derivative tracks rate of change.",
            "It captures motion at a single instant."
        ]
        self.setup_layout("The Derivative: Seeing the Slope", lecture_lines)
        
        # Assets
        microscope = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/microscope.svg", color=WHITE)
        speedometer = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg", color=WHITE)
        
        self.place_at_grid(microscope, "A6", scale_factor=0.5)
        self.add(microscope)
        
        # Curve Setup
        curve = FunctionGraph(lambda x: 0.1 * x**3 - 0.5 * x + 1, x_range=[-3, 3], color="#FF00FF")
        self.place_in_area(curve, 'C4', 'E6', scale_factor=1.2)
        
        point_marker = Dot(curve.point_from_proportion(0.5), color=YELLOW)
        self.place_at_grid(point_marker, 'D4', scale_factor=0.8)
        
        tangent_line = Line(start=[-1, 0, 0], end=[1, 0, 0], color=WHITE)
        self.place_at_grid(tangent_line, 'D4', scale_factor=1.0)
        
        # === Animation for Lecture Line 1 ===
        self.play(Create(curve), run_time=1)
        self.lecture[0].set_color("#00FF00")
        self.play(FadeIn(point_marker))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FFFF")
        self.play(curve.animate.scale(2, about_point=point_marker.get_center()), run_time=2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FFD700")
        self.place_at_grid(speedometer, "B6", scale_factor=0.5)
        self.play(Create(tangent_line), FadeIn(speedometer))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FF4500")
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#ADFF2F")
        self.play(FadeOut(curve), FadeOut(tangent_line), FadeOut(point_marker), FadeOut(microscope), FadeOut(speedometer))
        self.wait(1)
