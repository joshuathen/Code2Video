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
        lecture_lines = ["We use limits to find instantaneous change.", "Shrink the time interval toward zero.", "The secant line becomes a tangent."]
        self.setup_layout("The Method of Exhaustion (Limits)", lecture_lines)
        
        # Mobjects
        limit_text = MathTex(r"\lim_{\Delta t \to 0}", color=WHITE)
        curve = FunctionGraph(lambda x: 0.5 * x**2, x_range=[-2, 2], color=BLUE)
        
        # Grid setup
        self.place_at_grid(limit_text, "A3", scale_factor=1.5)
        self.place_in_area(curve, "C1", "E6", scale_factor=0.8)

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(limit_text), self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Create(curve), self.lecture[1].animate.set_color("#FFCC00"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        secant = Line(start=curve.point_from_proportion(0.3), end=curve.point_from_proportion(0.7), color=GREEN)
        self.play(Create(secant), self.lecture[2].animate.set_color("#00FF00"))
        
        # Tangent morphing
        tangent = Line(start=curve.point_from_proportion(0.48), end=curve.point_from_proportion(0.52), color=GREEN).scale(5)
        self.play(Transform(secant, tangent))
        self.wait(2)
