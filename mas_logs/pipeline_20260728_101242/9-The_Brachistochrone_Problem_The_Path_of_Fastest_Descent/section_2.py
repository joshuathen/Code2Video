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
        # Initialize the scene layout
        self.setup_layout("Prerequisite: Snell's Law and Fermat's Principle", [
            "Light always follows the path of least time.",
            "This is known as Fermat’s Principle of Least Time.",
            "Snell’s Law describes how light bends across mediums."
        ])

        # Colors
        color_ray = "#FFFF00"
        color_text = "#FFFFFF"
        color_formula = "#00FFFF"
        
        # === Animation for Lecture Line 1 ===
        # Show a light ray (#FFFF00) [Asset: light.svg] hitting a medium boundary [Asset: medium.svg].
        
        # Boundary line
        boundary = Line(self.grid["C1"], self.grid["C6"], color=BLUE_B)
        
        # Medium asset
        medium_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/medium.svg")
        self.place_at_grid(medium_icon, "C1", scale_factor=0.6)
        
        # Light asset
        light_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/light.svg")
        self.place_at_grid(light_icon, "B1", scale_factor=0.5)
        
        # Ray 1: from B1 (light source) to C3 (boundary)
        ray1 = Line(self.grid["B1"], self.grid["C3"], color=color_ray)
        # Ray 2: from C3 to E6
        ray2 = Line(self.grid["C3"], self.grid["E6"], color=color_ray)
        
        self.play(self.lecture[0].animate.set_color(color_ray))
        self.play(Create(boundary), FadeIn(medium_icon))
        self.play(FadeIn(light_icon))
        self.play(Create(ray1))
        self.play(Create(ray2))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # This is known as Fermat’s Principle of Least Time.
        self.play(self.lecture[1].animate.set_color(color_ray))
        
        # Fermat label - Fix: Issue 24 (place_in_area A4-A6)
        fermat_label = Text("Fermat's Principle", font_size=20, color=color_text)
        self.place_in_area(fermat_label, 'A4', 'A6')
        self.play(Write(fermat_label))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Snell’s Law describes how light bends across mediums.
        self.play(self.lecture[2].animate.set_color(color_formula))
        
        # Normal line for angles
        normal = DashedLine(self.grid["A3"], self.grid["F3"], color=GRAY_A)
        
        # theta1 label
        theta1_label = MathTex(r"\theta_1", color=color_text)
        self.place_at_grid(theta1_label, "B3", scale_factor=0.6)
        theta1_label.shift(LEFT * 0.4 + UP * 0.1)
        
        # theta2 label - Fix: Issue 25 (place_at_grid D4)
        theta2_label = MathTex(r"\theta_2", color=color_text)
        self.place_at_grid(theta2_label, 'D4', scale_factor=0.6)
        theta2_label.shift(LEFT * 0.3 + DOWN * 0.1) # Adjusted shift for new position
        
        # v1 label along ray 1
        v1_label = MathTex("v_1", color=color_text)
        self.place_at_grid(v1_label, "B2", scale_factor=0.8)
        
        # v2 label along ray 2
        v2_label = MathTex("v_2", color=color_text)
        self.place_at_grid(v2_label, "D5", scale_factor=0.8)
        
        # Snell's Law formula - Fix: Issue 26 (place_in_area E1-F4)
        snell_law = MathTex(r"\frac{\sin \theta_1}{v_1} = \frac{\sin \theta_2}{v_2} = \text{Constant}", color=color_formula)
        self.place_in_area(snell_law, 'E1', 'F4', scale_factor=0.8)
        
        self.play(Create(normal))
        self.play(
            Write(v1_label),
            Write(v2_label),
            Write(theta1_label),
            Write(theta2_label)
        )
        self.play(Write(snell_law))
        self.wait(5)
