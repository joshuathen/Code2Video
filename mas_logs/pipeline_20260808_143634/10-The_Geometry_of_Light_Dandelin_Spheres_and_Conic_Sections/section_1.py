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
            "Conic sections arise from cutting a double cone.",
            "Changing the cutting angle alters the shape produced.",
            "Shapes include circles, ellipses, parabolas, and hyperbolas."
        ]
        self.setup_layout("Prerequisite: The Cutting Plane", lecture_lines)
        
        # Load asset once
        cone_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg")

        # === Animation for Lecture Line 1 ===
        # Show a 3D double cone with a flat plane cutting it. (#FFFFFF)
        self.lecture[0].set_color("#FFFFFF")
        cone1 = self.place_at_grid(cone_icon.copy(), 'C2', scale_factor=1.5)
        plane1 = Line(start=np.array([-1, 0, 0]), end=np.array([1, 0, 0]), color=WHITE).shift(cone1.get_center())
        self.play(FadeIn(cone1), Create(plane1))

        # === Animation for Lecture Line 2 ===
        # Rotate the plane to change the intersection shape. (#FFFF00)
        self.lecture[1].set_color("#FFFF00")
        self.play(Rotate(plane1, angle=PI/4, about_point=cone1.get_center()))

        # === Animation for Lecture Line 3 ===
        # Display the resulting circle, ellipse, parabola, and hyperbola derived from the cone. (#00FF00)
        self.lecture[2].set_color("#00FF00")
        
        # Simple placeholders for conic sections
        c1 = Circle(radius=0.3, color=GREEN).set_stroke(width=2)
        e1 = Ellipse(width=0.6, height=0.3, color=GREEN).set_stroke(width=2)
        p1 = FunctionGraph(lambda x: x**2, x_range=[-0.5, 0.5], color=GREEN)
        h1 = VGroup(FunctionGraph(lambda x: 1/x, x_range=[0.2, 1], color=GREEN), 
                    FunctionGraph(lambda x: 1/x, x_range=[-1, -0.2], color=GREEN))
        
        conics = VGroup(c1, e1, p1, h1).arrange(RIGHT, buff=0.3)
        self.place_at_grid(conics, 'E3', scale_factor=0.8)
        
        self.play(FadeIn(conics))
        self.wait(2)
