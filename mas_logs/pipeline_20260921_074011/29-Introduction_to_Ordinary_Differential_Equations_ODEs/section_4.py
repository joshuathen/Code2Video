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
        self.setup_layout("The Role of Initial Conditions", [
            "ODE solutions form a family of curves.",
            "Initial conditions pin down one unique curve.",
            "They determine a specific path or state."
        ])
        
        # Setup visual elements
        axes = Axes(x_range=[-2, 2, 1], y_range=[-2, 2, 1], x_length=4, y_length=4)
        
        # Define curves for family
        curves = VGroup(*[
            axes.plot(lambda x: x**2 + c, color=GRAY)
            for c in [-1, -0.5, 0, 0.5, 1]
        ])
        
        area = VGroup(axes, curves)
        self.place_in_area(area, 'B3', 'F6', scale_factor=0.8)
        self.add(area)
        
        # Load assets
        pin = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/pin.svg")
        protractor = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/protractor.svg")
        
        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(curves))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        # Initial point and specific curve
        point = self.place_at_grid(pin.copy(), 'C4', scale_factor=0.2)
        specific_curve = axes.plot(lambda x: x**2, color="#FF0000")
        specific_curve.move_to(axes.c2p(0, 0) + np.array([0, 0.5, 0])) # Adjust to axis
        
        self.play(FadeIn(point))
        self.play(Create(specific_curve))
        self.lecture[1].set_color("#FF0000")

        # === Animation for Lecture Line 3 ===
        # Change initial condition
        new_point = self.place_at_grid(protractor.copy(), 'D5', scale_factor=0.2)
        new_curve = axes.plot(lambda x: x**2 + 0.5, color="#FFFF00")
        
        self.play(ReplacementTransform(point, new_point))
        self.play(ReplacementTransform(specific_curve, new_curve))
        self.lecture[2].set_color("#FFFF00")
        self.wait(2)
