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

class Section4Scene(TeachingScene):
    def construct(self):
        lecture_lines = [
            "Visualize the original curve alongside its derivative.",
            "The derivative's height equals the original slope.",
            "A roller coaster's slope defines its speed.",
            "The speedometer graph tracks the motion exactly.",
            "Comparing curves reveals their fundamental link."
        ]
        self.setup_layout("Visualizing the 'Derivative Function' Graph", lecture_lines)
        
        axes = Axes(x_range=[-2, 2], y_range=[-1, 5], axis_config={"include_tip": False}).scale(0.5)
        curve = axes.plot(lambda x: x**2, color=BLUE)
        deriv = axes.plot(lambda x: 2*x, color=YELLOW)
        
        coaster = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rollercoaster.svg").scale(0.5)
        speedo = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/speedometer.svg").scale(0.5)
        
        container = VGroup(axes, curve, deriv, coaster, speedo)
        self.place_in_area(container, 'B3', 'F6', scale_factor=0.95)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color(BLUE)
        self.play(FadeIn(coaster), FadeIn(axes), Create(curve))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color(YELLOW)
        self.play(Create(deriv))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color(RED)
        dot = Dot(color=RED).move_to(curve.point_from_proportion(0.5))
        self.play(FadeIn(dot))

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color(GREEN)
        self.play(FadeIn(speedo), dot.animate.move_to(curve.point_from_proportion(0.8)))

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color(PURPLE)
        self.play(FadeOut(dot))
