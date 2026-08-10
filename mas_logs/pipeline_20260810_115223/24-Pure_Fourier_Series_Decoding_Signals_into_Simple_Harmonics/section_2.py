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
            "Any periodic function can become an infinite sum.",
            "We add waves, increasing their frequency each step.",
            "Harmonics layer together to define complex shapes.",
            "Edges sharpen as we add more harmonic terms.",
            "The square wave emerges from these simple layers."
        ]
        self.setup_layout("The Fundamental Hypothesis", lecture_lines)
        
        # Define mobjects
        formula = MathTex(r"f(x) = \sum_{n=1}^{\infty} A_n \sin(nx)", font_size=36)
        square_vis = Square(side_length=2.0, color="#FFFF00", fill_opacity=0.2)
        
        # Dummy assets
        # Note: Storyboard asked for these icons, though they are path to "/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg"
        asset_formula = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        asset_square = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg")
        
        formula_label = VGroup(formula, asset_formula).arrange(RIGHT)

        # Animation loop
        for i, line in enumerate(self.lecture):
            # Apply color to current line
            if i == 0:
                self.play(line.animate.set_color("#FFFFFF"), run_time=0.5)
            elif i == 1:
                self.play(line.animate.set_color("#00FF00"), run_time=0.5)
            elif i == 2:
                self.play(line.animate.set_color("#00FFFF"), run_time=0.5)
            elif i == 3:
                self.play(line.animate.set_color("#FF00FF"), run_time=0.5)
            elif i == 4:
                self.play(line.animate.set_color("#FFFF00"), run_time=0.5)
            
            # Animation for Lecture Line 1
            if i == 0:
                self.place_at_grid(formula_label, 'C3', scale_factor=0.9)
                self.play(Write(formula_label))
            
            # Animation for Lecture Line 2
            elif i == 1:
                self.play(formula.animate.set_color("#00FF00"), run_time=0.5)
            
            # Animation for Lecture Line 3
            elif i == 2:
                self.play(formula.animate.set_color("#00FFFF"), run_time=0.5)
            
            # Animation for Lecture Line 4
            elif i == 3:
                self.play(formula.animate.set_color("#FF00FF"), run_time=0.5)
            
            # Animation for Lecture Line 5
            elif i == 4:
                self.place_in_area(square_vis, 'D4', 'F6', scale_factor=0.7)
                self.play(Create(square_vis))
                self.play(formula.animate.set_color("#FFFF00"), run_time=0.5)
        
        self.wait(2)
