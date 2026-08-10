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
        self.setup_layout("Prerequisites: The Cone and the Plane", 
                          ["Conic sections result from plane-cone intersections.", 
                           "Imagine a flashlight creating shapes on a wall.", 
                           "The intersection angle determines the specific shape."])
        
        # Define objects
        cone = VGroup(
            Line(UP, DOWN),
            Line(LEFT, RIGHT).shift(UP*0.5),
            Line(LEFT, RIGHT).shift(DOWN*0.5)
        ).set_color(WHITE)
        
        plane = Polygon(
            np.array([-1, 0.5, 0]), np.array([1, 0.5, 0]),
            np.array([1.5, -0.5, 0]), np.array([-1.5, -0.5, 0]),
            color=BLUE, fill_opacity=0.3
        )
        
        ellipse = Ellipse(width=0.8, height=0.4, color="#FF00FF")
        
        # Load asset
        flashlight = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/flashlight.svg")

        # === Animation for Lecture Line 1 ===
        self.play(FadeIn(self.lecture[0]))
        self.lecture[0].set_color(BLUE) # Changed color for visual indicator
        self.place_at_grid(cone, 'B4', scale_factor=0.8)
        self.play(Create(cone))

        # === Animation for Lecture Line 2 ===
        self.play(FadeIn(self.lecture[1]))
        self.lecture[1].set_color(BLUE)
        self.place_at_grid(plane, 'D4', scale_factor=0.9)
        self.place_at_grid(flashlight, 'D6', scale_factor=0.5)
        self.play(FadeIn(plane), FadeIn(flashlight))

        # === Animation for Lecture Line 3 ===
        self.play(FadeIn(self.lecture[2]))
        self.lecture[2].set_color(BLUE)
        self.place_in_area(ellipse, 'C3', 'E5', scale_factor=0.8)
        self.play(Create(ellipse))

        self.wait(2)
