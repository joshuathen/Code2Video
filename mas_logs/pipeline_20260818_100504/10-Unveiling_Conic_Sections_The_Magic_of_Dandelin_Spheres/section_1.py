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
        self.setup_layout("Prerequisite: The Conic Anatomy", ["Meet the cone and the slicing plane.", "Intersection creates different conic shapes.", "Why do these shapes appear?"])
        
        # Create objects
        # Use SVG for the cone as requested in Issue 16
        cone = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/cone.svg").set_color(WHITE)
        self.place_in_area(cone, 'B2', 'E5', scale_factor=1.0)
        
        base_circle = Circle(radius=0.5, color=YELLOW)
        base_circle.move_to(cone.get_bottom())
        
        vertex_label = Text("V", color=BLUE).scale(0.5)
        
        plane = Polygon(np.array([-2, 0, 1]), np.array([2, 0, 1]), np.array([2, 0, -1]), np.array([-2, 0, -1]), color=RED)
        plane.set_fill(RED, opacity=0.3)
        
        intersection = Ellipse(width=0.7, height=0.5, color=GREEN)
        
        # === Animation for Lecture Line 1 ===
        # Meet the cone and the slicing plane.
        self.play(FadeIn(cone))
        self.lecture[0].set_color(YELLOW)
        self.play(Create(base_circle))
        # Place vertex_label as per Issue 21
        self.place_at_grid(vertex_label, 'A6', scale_factor=0.7)
        self.play(Write(vertex_label))

        # === Animation for Lecture Line 2 ===
        # Intersection creates different conic shapes.
        self.lecture[1].set_color(GREEN)
        self.play(Create(plane))
        
        # === Animation for Lecture Line 3 ===
        # Why do these shapes appear?
        self.lecture[2].set_color(BLUE)
        # Place intersection as per Issue 22
        self.place_in_area(intersection, 'C3', 'D4', scale_factor=0.9)
        self.play(Create(intersection))
        self.wait(2)
