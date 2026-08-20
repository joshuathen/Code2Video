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
        self.setup_layout("The Mechanism: The Power of Aggregation", [
            "Aggregation reveals the hidden structure.",
            "Averages smooth out extreme values.",
            "Many averages form a Bell Curve."
        ])
        
        # Assets
        beaker = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/beaker.svg")
        scale_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/scale.svg")
        calc = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/calculator.svg")
        
        # Objects
        circles = VGroup(*[Circle(radius=0.3, fill_opacity=0.6) for _ in range(3)])
        beaker.set_color("#1ABC9C")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(circles.arrange(RIGHT, buff=0.2)), FadeIn(beaker))
        self.place_at_grid(VGroup(circles, beaker), 'B4', scale_factor=0.6)
        self.play(circles.animate.set_color("#1ABC9C"))
        self.lecture[0].set_color("#1ABC9C")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        avg_indicator = Square(side_length=0.5, color="#2980B9", fill_opacity=0.8)
        scale_icon.set_color("#2980B9")
        self.place_at_grid(avg_indicator, 'D4', scale_factor=0.7)
        self.place_at_grid(scale_icon, 'C3', scale_factor=0.5)
        self.play(FadeIn(avg_indicator), FadeIn(scale_icon))
        self.lecture[1].set_color("#2980B9")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        bell = FunctionGraph(lambda x: np.exp(-x**2), x_range=[-2, 2], color="#F39C12")
        calc.set_color("#F39C12")
        self.place_in_area(bell, 'D3', 'F6', scale_factor=0.6)
        self.place_at_grid(calc, 'E2', scale_factor=0.5)
        self.play(Create(bell), FadeIn(calc))
        self.lecture[2].set_color("#F39C12")
        self.wait(2)
