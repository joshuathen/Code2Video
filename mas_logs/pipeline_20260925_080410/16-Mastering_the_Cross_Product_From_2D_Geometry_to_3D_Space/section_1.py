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
        self.setup_layout("Prerequisites: Vectors and Planes", [
            "Vectors have both magnitude and direction.",
            "Planes are flat surfaces in space.",
            "The Right-Hand Rule defines orthogonal direction."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Vector visualization (v1, v2)
        v1 = Arrow(start=ORIGIN, end=RIGHT*1.5+UP*0.5, color=WHITE)
        v2 = Arrow(start=ORIGIN, end=RIGHT*1.0-UP*0.8, color=WHITE)
        v_group = VGroup(v1, v2)
        self.place_at_grid(v_group, 'C2', scale_factor=0.7)
        self.play(Create(v_group), self.lecture[0].animate.set_color(WHITE))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Plane visualization
        plane = Rectangle(width=2.5, height=1.5, color="#808080", fill_opacity=0.3)
        self.place_in_area(plane, 'D3', 'E4', scale_factor=0.8)
        self.play(FadeIn(plane), self.lecture[1].animate.set_color("#808080"))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Right-Hand Rule visualization
        right_hand_rule = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/hand.svg")
        right_hand_rule.set_color("#FF00FF")
        self.place_at_grid(right_hand_rule, 'F3', scale_factor=0.6)
        
        # Normal vector (orthogonal)
        normal = Arrow(start=ORIGIN, end=UP*1.5, color="#FF00FF")
        normal.move_to(plane.get_center() + UP*0.5)
        
        self.play(
            Create(right_hand_rule),
            Create(normal),
            self.lecture[2].animate.set_color("#FF00FF")
        )
        self.wait(2)
