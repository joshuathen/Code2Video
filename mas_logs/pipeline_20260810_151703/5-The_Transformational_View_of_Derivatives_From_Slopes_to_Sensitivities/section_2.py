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
        lecture_lines = [
            "Imagine a magnifying glass on a curve.",
            "Shrink the interval towards zero.",
            "The secant line becomes a tangent.",
            "This is the derivative's geometric form.",
            "Zooming reveals the curve's instant slope."
        ]
        self.setup_layout("The Transformation: From Secant to Tangent", lecture_lines)
        
        # Define curve and objects
        curve = FunctionGraph(lambda x: 0.1 * x**2, x_range=[-4, 4], color=BLUE)
        self.place_in_area(curve, "B2", "D5", scale_factor=0.8)
        
        # Magnifying glass asset
        mag_glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/magnifying-glass.svg")
        self.place_at_grid(mag_glass, "B4", scale_factor=0.5)
        
        p1 = np.array([-1, 0, 0])
        p2 = np.array([1, 0.2, 0])
        
        secant = Line(p1, p2, color=WHITE)
        dx_label = MathTex(r"\\Delta x", color="#00FFFF")
        self.place_at_grid(dx_label, "E4", scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color(BLUE))
        self.play(Create(curve), FadeIn(mag_glass))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[0].animate.set_color(WHITE), self.lecture[1].animate.set_color("#00FFFF"))
        self.play(Create(secant), FadeIn(dx_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[1].animate.set_color(WHITE), self.lecture[2].animate.set_color("#FFFF00"))
        tangent = Line(np.array([-0.5, 0.1, 0]), np.array([0.5, 0.1, 0]), color="#FFFF00")
        tangent.move_to(curve.point_from_proportion(0.5))
        self.play(Transform(secant, tangent), FadeOut(dx_label), FadeOut(mag_glass))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[2].animate.set_color(WHITE), self.lecture[3].animate.set_color(GREEN))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[3].animate.set_color(WHITE), self.lecture[4].animate.set_color(ORANGE))
        self.play(curve.animate.scale(1.2), run_time=2)
        self.wait(2)
