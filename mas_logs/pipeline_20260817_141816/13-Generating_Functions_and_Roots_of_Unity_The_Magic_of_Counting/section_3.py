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
        self.setup_layout("Applying the Complex Tool", [
            "Substitute roots of unity into f(x).",
            "Consider the example f(x) = (1+x)^n.",
            "We want sums divisible by 3.",
            "Calculate (1/3) * sum(1 + omega^j)^n.",
            "This extracts the desired combinatorial counts."
        ])
        
        # Assets
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg").set_color("#00BFFF")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg").set_color("#32CD32")
        
        # Elements for animation
        vector = Arrow(ORIGIN, RIGHT * 2, color="#FF6347")
        z_variable = Text("z", font_size=24, color="#FF6347")
        
        # Place based on review feedback
        self.place_at_grid(vector, 'E4', scale_factor=0.8)
        self.place_at_grid(z_variable, 'B5', scale_factor=1.0)
        
        complex_formula_group = VGroup(compass, protractor)
        self.place_in_area(complex_formula_group, 'D3', 'F5', scale_factor=0.9)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#FF6347"))
        
        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#00BFFF"))
        self.play(Rotate(vector, angle=PI/3, about_point=self.grid['E4']))
        
        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#32CD32"))
        
        # === Animation for Lecture Line 4 ===
        self.play(self.lecture[3].animate.set_color("#FFD700"))
        self.play(vector.animate.scale(0.8))
        
        # === Animation for Lecture Line 5 ===
        self.play(self.lecture[4].animate.set_color("#FF69B4"))
        self.wait(1)
