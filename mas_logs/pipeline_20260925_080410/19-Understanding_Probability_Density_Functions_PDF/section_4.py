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
        self.setup_layout("Real-World Application: The Squirrel’s Acorn Search", [
            "PDF predicts outcomes like squirrel acorn drops.", 
            "Curve spikes show where events likely occur.", 
            "Area under spikes reveals hunting success zones."
        ])
        
        # Elements
        axes = Axes(x_range=[-3, 3, 1], y_range=[0, 3, 1], x_length=5, y_length=3)
        self.place_at_grid(axes, 'E3', scale_factor=0.8)
        
        # PDF curve
        pdf = axes.plot(lambda x: 2.5 * np.exp(-x**2), color="#33FF57")
        
        # Load Assets
        squirrel = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/squirrel.svg")
        acorn = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/acorn.svg")
        
        # Position squirrel (Fixes #28)
        self.place_at_grid(squirrel, 'D4', scale_factor=0.3)
        # Position tree-like object (Fixes #29)
        self.place_at_grid(acorn, 'B3', scale_factor=0.3)

        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#33FF57"), Create(acorn), Write(axes))
        self.play(Create(pdf))

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FF5733"), squirrel.animate.move_to(axes.c2p(0, 0)))
        self.play(squirrel.animate.move_to(axes.c2p(0.5, 2)))

        # === Animation for Lecture Line 3 ===
        area = axes.get_area(pdf, x_range=[-1, 1], color="#FF5733", opacity=0.3)
        self.play(self.lecture[2].animate.set_color("#FF5733"), Create(area))
        self.wait(2)
