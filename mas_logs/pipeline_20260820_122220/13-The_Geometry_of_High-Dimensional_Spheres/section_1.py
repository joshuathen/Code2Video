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
        lecture_lines = [
            "Pythagoras works perfectly in two dimensions.",
            "Extending to three, we simply add a Z-term.",
            "Generalize this to n-dimensions using n coordinates.",
            "The distance formula naturally scales to any dimension.",
            "Thus, we define spheres in n-dimensional space."
        ]
        self.setup_layout("Introduction: Beyond 3D Space", lecture_lines)
        
        # Mobjects
        # Using SVGMobject for asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg
        circle = Circle(radius=0.6, color="#BDC3C7")
        axes_2d = Axes(x_range=[-2, 2], y_range=[-2, 2], axis_config={"include_tip": False}, color="#BDC3C7").scale(0.3)
        axes_3d = ThreeDAxes(x_range=[-2, 2], y_range=[-2, 2], z_range=[-2, 2], axis_config={"include_tip": False}, color="#BDC3C7").scale(0.3)
        formula = MathTex(r"d = \sqrt{\sum_{i=1}^n x_i^2}", color="#E74C3C")
        hypersphere = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg", color="#3498DB")
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#BDC3C7"))
        self.place_at_grid(axes_2d, "C5", scale_factor=0.8)
        self.play(Create(axes_2d))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#BDC3C7"))
        self.place_at_grid(axes_3d, "D5", scale_factor=0.8)
        self.play(FadeOut(axes_2d), Create(axes_3d))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#E74C3C"))
        self.place_at_grid(formula, "C5", scale_factor=0.8)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#E74C3C"))
        self.wait(1)

        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#3498DB"))
        self.place_at_grid(hypersphere, "E5", scale_factor=0.8)
        self.play(FadeOut(formula), FadeIn(hypersphere))
        self.wait(1)
