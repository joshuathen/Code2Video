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
            "Square the integral for a 2D surface.",
            "Transition from Cartesian to polar coordinates.",
            "Polar coordinates naturally involve circles and π."
        ]
        self.setup_layout("The Coordinate Transformation Trick", lecture_lines)
        
        # Mobjects
        gaussian_x = MathTex(r"I = \int_{-\infty}^{\infty} e^{-x^2} dx")
        gaussian_y = MathTex(r"I = \int_{-\infty}^{\infty} e^{-y^2} dy")
        combined = MathTex(r"I^2 = \iint_{\mathbb{R}^2} e^{-(x^2+y^2)} dx dy")
        polar = MathTex(r"I^2 = \int_{0}^{2\pi} \int_{0}^{\infty} e^{-r^2} r dr d\theta")
        
        # Assets
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        
        # Setup visual elements
        self.place_at_grid(gaussian_x, 'B2', scale_factor=0.8)
        self.place_at_grid(gaussian_y, 'B5', scale_factor=0.8)
        self.place_at_grid(protractor, 'D2', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FFFFFF"))
        self.play(Write(combined))
        self.place_at_grid(combined, 'D2', scale_factor=0.8)
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFD700"))
        self.play(FadeOut(gaussian_x), FadeOut(gaussian_y), FadeOut(protractor), combined.animate.set_color("#00BFFF"))
        self.place_at_grid(compass, 'D2', scale_factor=0.5)
        self.play(ReplacementTransform(combined, polar))
        self.place_at_grid(polar, 'D5', scale_factor=0.7)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#FF4500"))
        # Highlighting the π in the polar integral
        # Note: MathTex objects structure their sub-mobjects. 
        # In this string: I^2 = \int_{0}^{2\pi} ...
        # Polar subobjects: 0:'I', 1:'^', 2:'2', 3:'=', 4:'\int', 5:'{0}', 6:'{2\pi}', ...
        # We target the π part.
        circle = Circle(radius=0.2, color=WHITE).move_to(polar[0][6].get_center())
        self.play(Create(circle))
        self.wait(2)
