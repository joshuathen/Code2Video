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
        self.setup_layout("Defining Refraction and Snell's Law", [
            "Refraction occurs when light changes direction between media.",
            "Snell's Law predicts this change mathematically.",
            "We measure angles relative to the normal line."
        ])
        
        # Elements for animation
        laser = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/laser.svg")
        glass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/glass.svg")
        
        interface = Line(start=self.grid["C1"], end=self.grid["C6"], color=GRAY)
        normal = DashedLine(start=self.grid["A3"], end=self.grid["E3"], color=WHITE)
        
        incident_ray = Line(start=self.grid["A1"], end=self.grid["C3"], color=BLUE)
        refracted_ray = Line(start=self.grid["C3"], end=self.grid["E5"], color=RED)
        
        # Group diagram to satisfy critic
        refraction_diagram = VGroup(interface, normal, incident_ray, refracted_ray, laser, glass)
        self.place_in_area(refraction_diagram, 'B3', 'D6', scale_factor=0.9)
        
        snell_formula = MathTex(r"n_1 \sin\theta_1 = n_2 \sin\theta_2", color="#FFFF00")
        self.place_in_area(snell_formula, 'E2', 'F5', scale_factor=0.8)
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(interface), FadeIn(normal), FadeIn(laser))
        self.play(Create(incident_ray))
        self.lecture[0].set_color(BLUE)

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(glass))
        self.play(Create(refracted_ray))
        self.lecture[1].set_color(RED)
        
        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(snell_formula))
        self.lecture[2].set_color("#FFFF00")
        
        self.wait(2)
