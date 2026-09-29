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
        self.setup_layout("The Mathematical Trick: Switching to Polar Coordinates", 
                          ["Solve I squared to integrate.", 
                           "Combine the two integrals.", 
                           "Switch to polar coordinates now.", 
                           "Coordinate transformation makes it easier.", 
                           "The geometry becomes much simpler."])
        
        # === Animation for Lecture Line 1 ===
        # Load asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg
        grid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_in_area(grid_icon, "A5", "B6", scale_factor=0.3)
        formula1 = MathTex(r"I^2 = \left(\int e^{-x^2} dx\right)^2", color="#FFFFFF")
        self.place_in_area(formula1, "A2", "B5", scale_factor=0.8)
        self.play(Write(formula1), FadeIn(grid_icon))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        formula2 = MathTex(r"I^2 = \iint e^{-(x^2+y^2)} dx dy", color="#7FFF00")
        self.place_in_area(formula2, "A2", "B5", scale_factor=0.8)
        self.play(ReplacementTransform(formula1, formula2))
        self.lecture[1].set_color("#7FFF00")

        # === Animation for Lecture Line 3 ===
        polar_text = MathTex(r"x^2+y^2=r^2, \quad dA=r dr d\theta", color="#FFD700")
        self.place_in_area(polar_text, "C1", "D3", scale_factor=0.6)
        self.play(Write(polar_text))
        self.lecture[2].set_color("#FFD700")

        # === Animation for Lecture Line 4 ===
        polar_grid = PolarPlane(size=4, azimuth_step=8).scale(0.5)
        self.place_in_area(polar_grid, "D3", "F6", scale_factor=0.6)
        self.play(Create(polar_grid))
        self.play(Flash(polar_grid, color="#FFD700"))
        self.lecture[3].set_color("#FFD700")

        # === Animation for Lecture Line 5 ===
        # Load asset /scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg
        protractor_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        self.place_in_area(protractor_icon, "C5", "C6", scale_factor=0.3)
        
        r_label = MathTex("r", color="#FF69B4")
        theta_label = MathTex(r"\theta", color="#FF69B4")
        self.place_at_grid(r_label, "D5", scale_factor=0.7)
        self.place_at_grid(theta_label, "E6", scale_factor=0.7)
        
        self.play(FadeIn(protractor_icon), FadeIn(r_label), FadeIn(theta_label))
        self.lecture[4].set_color("#FF69B4")
