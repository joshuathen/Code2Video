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
            "Real convergence: distance shrinks towards a target.",
            "Imagine a squirrel hopping towards an acorn.",
            "Distances like 1/2, 1/4, 1/8 steadily decrease."
        ]
        self.setup_layout("The Familiar World: Real Convergence", lecture_lines)
        
        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        number_line = NumberLine(
            x_range=[0, 2.5, 0.5], 
            length=4, 
            include_numbers=True,
            font_size=24
        )
        self.place_in_area(number_line, "C2", "C5", scale_factor=0.9)
        self.play(FadeIn(number_line))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF00")
        squirrel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/squirrel.svg")
        self.place_at_grid(squirrel, "B2", scale_factor=0.3)
        
        label_1unit = Text("1 unit", font_size=18, color="#FF0000")
        self.place_at_grid(label_1unit, "B3", scale_factor=0.7)
        
        self.play(FadeIn(squirrel), FadeIn(label_1unit))

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#FF00FF")
        acorn = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/acorn.svg")
        self.place_at_grid(acorn, "D4", scale_factor=0.3)
        
        ball_label = Text("Target", font_size=18, color="#FFFF00")
        self.place_at_grid(ball_label, "D3", scale_factor=0.8)
        
        self.play(FadeIn(acorn), FadeIn(ball_label))
        
        # Animate squirrel movement
        self.play(squirrel.animate.move_to(number_line.n2p(1.25)), run_time=1.0)
        
        self.wait(1)
