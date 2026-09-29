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
            "Total angle rotated is the key invariant.",
            "Each pivot switch adds a positive angle.",
            "Rotation is always counter-clockwise."
        ]
        self.setup_layout("Prerequisite: Angles and Slopes", lecture_lines)
        
        # Load assets
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        compass = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        
        # Define objects
        line1 = Line(start=LEFT*1.5, end=RIGHT*1.5, color="#00FFFF")
        line2 = Line(start=DOWN*1.2 + LEFT*0.4, end=UP*1.2 + RIGHT*0.4, color="#00FFFF")
        angle = Angle(line1, line2, radius=0.6, color="#FFFF00")
        formula = MathTex(r"\\Delta \\theta > 0", color="#FF00FF")

        # Layout group
        animation_group = VGroup(line1, line2, angle, protractor, compass, formula)
        self.place_in_area(animation_group, 'B1', 'E3', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.play(Create(line1), Create(line2))
        self.place_at_grid(protractor, 'B2', scale_factor=0.3)
        self.play(FadeIn(protractor))
        self.lecture[0].set_color("#00FFFF")

        # === Animation for Lecture Line 2 ===
        self.place_at_grid(angle, 'B2', scale_factor=0.8)
        self.play(Create(angle))
        self.place_at_grid(compass, 'C2', scale_factor=0.3)
        self.play(FadeIn(compass))
        self.play(Rotate(line2, angle=PI/6, about_point=line2.get_center()))
        self.lecture[1].set_color("#FFFF00")

        # === Animation for Lecture Line 3 ===
        self.place_at_grid(formula, 'E2', scale_factor=0.7)
        self.play(Write(formula))
        self.lecture[2].set_color("#FF00FF")
        
        self.wait(2)
