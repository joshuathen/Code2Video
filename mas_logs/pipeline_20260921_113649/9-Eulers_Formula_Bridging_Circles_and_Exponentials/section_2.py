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
            "Taylor series approximate complex functions as polynomials.",
            "Summing e to the x terms creates curves.",
            "Plugging 'ix' into the series reveals patterns.",
            "The series separates into real and imaginary parts.",
            "These parts reveal sine and cosine relationships."
        ]
        self.setup_layout("Visualizing Taylor Series (The Bridge)", lecture_lines)
        
        # Setup Axes
        axes = Axes(x_range=[-3, 3], y_range=[-2, 4], axis_config={"include_tip": False})
        self.place_in_area(axes, 'E1', 'F6', scale_factor=0.35)
        
        # Assets
        bridge_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/bridge.svg")
        self.place_at_grid(bridge_icon, 'B3', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        # Display a function curve (f(x) = e^x)
        exp_curve = axes.plot(lambda x: np.exp(x), color="#FFFFFF")
        self.play(FadeIn(bridge_icon), Create(exp_curve), self.lecture[0].animate.set_color("#FFFFFF"))

        # === Animation for Lecture Line 2 ===
        # Overlay Taylor polynomial approximation for e^x
        taylor_poly = axes.plot(lambda x: 1 + x + x**2/2 + x**3/6, color="#FFFF00")
        self.play(Create(taylor_poly), self.lecture[1].animate.set_color("#FFFF00"))

        # === Animation for Lecture Line 3, 4, 5 ===
        # Show series terms adding up to the curve
        better_poly = axes.plot(lambda x: 1 + x + x**2/2 + x**3/6 + x**4/24, color="#00FFFF")
        self.play(
            ReplacementTransform(taylor_poly, better_poly),
            self.lecture[2].animate.set_color("#00FFFF"),
            self.lecture[3].animate.set_color("#00FFFF"),
            self.lecture[4].animate.set_color("#00FFFF")
        )
        self.wait(2)
