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
        self.setup_layout("Application: The Jacobian and Multivariable Intuition", 
                          ["Jacobians extend transformations to 2D.", 
                           "Squares map to parallelograms.", 
                           "Robot arms trace paths beautifully."])
        
        # --- Animation Content ---
        robot = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/robot.svg")
        
        # === Animation for Lecture Line 1 ===
        # Show a 2D to 2D vector field transformation. (#FFFFFF)
        self.lecture[0].set_color("#FFFFFF")
        
        square = Square(side_length=1.5, color=BLUE)
        self.place_in_area(square, "B2", "D4")
        
        self.place_at_grid(robot, "B5", scale_factor=0.5)
        
        self.play(FadeIn(square), FadeIn(robot))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Represent the Jacobian matrix as a set of basis vectors. (#F1C40F)
        self.lecture[1].set_color("#F1C40F")
        
        basis_i = Arrow(start=ORIGIN, end=RIGHT*1.2, color=RED)
        basis_j = Arrow(start=ORIGIN, end=UP*1.2, color=GREEN)
        basis_group = VGroup(basis_i, basis_j)
        self.place_at_grid(basis_group, 'C5', scale_factor=0.7)
        
        self.play(Create(basis_group))
        
        # Transformation of square to parallelogram
        new_square = Polygon([-0.5, -0.7, 0], [1.0, -0.5, 0], [0.5, 1.2, 0], [-1.0, 1.0, 0], color=YELLOW)
        self.place_in_area(new_square, 'B4', 'D6', scale_factor=0.6)
        
        self.play(Transform(square, new_square))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Demonstrate the local stretching and rotation of a small patch. (#2ECC71)
        self.lecture[2].set_color("#2ECC71")
        
        circle = Circle(radius=0.5, color=WHITE).set_stroke(width=2)
        self.place_at_grid(circle, 'E3', scale_factor=0.8)
        
        self.play(Create(circle))
        self.play(Rotating(circle, angle=2*PI, run_time=2))
        self.wait(2)
