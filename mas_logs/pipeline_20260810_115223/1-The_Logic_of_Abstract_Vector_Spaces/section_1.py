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
        self.setup_layout("Prerequisite Review: From Arrows to Rules", [
            "Vectors are arrows with length and direction.",
            "They follow specific algebraic rules.",
            "Addition and scaling are fundamental."
        ])
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        map_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/map.svg")

        # === Animation for Lecture Line 1 ===
        arrow = Vector(direction=RIGHT*1.5 + UP*1.5, color="#FF5733")
        self.place_at_grid(arrow, 'B4', scale_factor=0.6)
        self.place_at_grid(compass, 'B2', scale_factor=0.5)
        
        self.play(Create(arrow), FadeIn(compass))
        self.lecture[0].set_color("#FF5733")

        # === Animation for Lecture Line 2 ===
        # Animate vector stretching and rotating
        self.play(
            arrow.animate.scale(1.5).rotate(PI/4),
            run_time=2
        )
        arrow.set_color("#33FF57")
        self.lecture[1].set_color("#33FF57")

        # === Animation for Lecture Line 3 ===
        dot = Dot(color="#3357FF")
        self.place_at_grid(dot, 'B5', scale_factor=0.5)
        self.place_at_grid(map_icon, 'C5', scale_factor=0.5)
        
        self.play(
            FadeOut(arrow),
            FadeOut(compass),
            FadeIn(dot),
            FadeIn(map_icon),
            run_time=2
        )
        self.lecture[2].set_color("#3357FF")
        self.wait(2)
