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
        self.setup_layout("Prerequisite: The Static Slope", ["Slope measures change between two static points.", "Rise over run determines the steepness.", "This describes a steady mountain trail."])
        
        # Create elements
        axes = NumberPlane(x_range=[-1, 3, 1], y_range=[-1, 5, 1], x_length=3, y_length=3)
        curve = axes.plot(lambda x: x**2, x_range=[-1, 2.2], color="#3498DB")
        formula = MathTex("f(x)=x^2", color="#FFFFFF")
        
        mountain = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/mountain.svg", color=WHITE)
        trail = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/trail.svg", color=WHITE)
        
        x1, x2 = 0.5, 1.8
        p1 = Dot(axes.c2p(x1, x1**2), color="#E74C3C")
        p2 = Dot(axes.c2p(x2, x2**2), color="#E74C3C")
        secant = Line(p1.get_center(), p2.get_center(), color="#2ECC71")
        
        graph_group = VGroup(axes, curve, p1, p2, secant)
        
        # Apply layout constraints per critic suggestions
        self.place_in_area(axes, 'B3', 'E6', scale_factor=0.5)
        self.place_at_grid(formula, 'B2', scale_factor=0.7)
        self.place_in_area(graph_group, 'C4', 'F6', scale_factor=0.6)
        
        # Positioning assets
        self.place_at_grid(mountain, 'A4', scale_factor=0.4)
        self.place_at_grid(trail, 'D4', scale_factor=0.4)

        # === Animation for Lecture Line 1 ===
        self.play(Create(curve), Write(formula), FadeIn(mountain))
        self.lecture[0].set_color("#3498DB")
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(Create(p1), Create(p2))
        self.lecture[1].set_color("#E74C3C")
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(Create(secant), FadeIn(trail))
        self.lecture[2].set_color("#2ECC71")
        self.wait(2)
