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
        self.setup_layout("Prerequisite: The Average Rate of Change", [
            "Average rate is the slope of a secant line.",
            "It measures change over an interval of time.",
            "Think of speed over a distance."
        ])
        
        # Setup graph in B2-C5
        axes = Axes(x_range=[0, 5], y_range=[0, 5], axis_config={"include_tip": False})
        curve = axes.plot(lambda x: 0.2 * x**2 + 1, color=WHITE)
        group = VGroup(axes, curve)
        self.place_in_area(group, 'B2', 'C5', scale_factor=0.6)
        
        a, h = 1.0, 2.0
        p1 = axes.c2p(a, 0.2 * a**2 + 1)
        p2 = axes.c2p(a + h, 0.2 * (a + h)**2 + 1)
        
        dot1 = Dot(p1, color="#FF00FF")
        dot2 = Dot(p2, color="#FF00FF")
        secant = Line(p1, p2, color="#00FFFF")
        
        # Assets
        car = ImageMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/car.png")
        speedo = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg")
        
        # Labels
        car.scale(0.3).next_to(dot1, UP, buff=0.1)
        speedo.scale(0.3).next_to(dot2, UP, buff=0.1)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FF00FF")
        self.play(FadeIn(dot1), FadeIn(dot2), Create(secant), FadeIn(car))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FFFFFF")
        formula = MathTex(r"\frac{f(a+h) - f(a)}{h}", color="#FFFFFF")
        self.place_at_grid(formula, 'D3', scale_factor=0.7)
        self.play(Write(formula))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.play(FadeIn(speedo))
        self.wait(2)
