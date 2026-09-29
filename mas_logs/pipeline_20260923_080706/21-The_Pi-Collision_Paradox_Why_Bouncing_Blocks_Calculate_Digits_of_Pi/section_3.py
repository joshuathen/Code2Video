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
        self.setup_layout("The Geometric Mapping", [
            "We map velocity space into a circular geometry.",
            "Elastic collisions become billiard ball reflections in a circle.",
            "The total arc length equals the digits of Pi."
        ])
        
        # Elements
        circle = Circle(radius=1.5, color=BLUE)
        billiard_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/billiard.svg", color=WHITE)
        
        geometry_group = VGroup(circle, billiard_icon)
        self.place_in_area(geometry_group, 'B4', 'E6', scale_factor=0.5)
        
        # === Animation for Lecture Line 1 ===
        self.play(self.lecture[0].animate.set_color("#F1C40F"))
        self.play(Create(circle))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        self.play(self.lecture[1].animate.set_color("#FFFFFF"))
        path = VMobject(color=RED)
        path.set_points_smoothly([
            geometry_group.get_center() + np.array([-1.2, 0.9, 0]),
            geometry_group.get_center() + np.array([0.5, -1.2, 0]),
            geometry_group.get_center() + np.array([1.3, 0.4, 0]),
            geometry_group.get_center() + np.array([-0.8, -1.1, 0])
        ])
        self.play(FadeIn(billiard_icon), Create(path), run_time=2)
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        self.play(self.lecture[2].animate.set_color("#E67E22"))
        arc = Arc(radius=1.5, start_angle=PI/4, angle=PI/2, color="#E67E22", stroke_width=8)
        self.place_at_grid(arc, 'C5', scale_factor=0.6)
        self.play(Create(arc))
        self.wait(1)
