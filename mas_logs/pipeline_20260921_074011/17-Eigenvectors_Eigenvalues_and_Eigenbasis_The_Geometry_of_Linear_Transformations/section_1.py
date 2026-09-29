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
            "Linear transformations stretch or rotate space.",
            "Most vectors change their direction.",
            "Special vectors stay on their span.",
            "These are the eigen-directions.",
            "Visualize stretching rubber along an axis."
        ]
        self.setup_layout("The Intuition: Stretching vs. Rotating", lecture_lines)
        
        # Assets
        rubber_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/rubber.svg")
        
        # Setup base objects
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_numbers": False}).scale(0.6)
        self.place_in_area(axes, 'C2', 'F6', scale_factor=0.65)
        self.add(axes)
        
        # Add rubber icon to axes
        self.place_in_area(rubber_icon, "A1", "A3", scale_factor=0.3)
        self.add(rubber_icon)

        # 1. Rotating vector
        vec_rot = Vector([1, 1], color="#FFFFFF")
        self.place_at_grid(vec_rot, 'D3', scale_factor=0.9)
        
        # 2. Stretching vector
        vec_stretch = Vector([1, 0], color="#FF00FF")
        self.place_at_grid(vec_stretch, 'E4', scale_factor=0.9)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#FFFFFF")
        self.play(Create(axes), FadeIn(rubber_icon), run_time=1)

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#FF00FF")
        self.add(vec_rot)
        self.play(Rotate(vec_rot, angle=PI, about_point=axes.c2p(0, 0)), run_time=2)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FFFF")
        self.add(vec_stretch)
        # Add eigen-line visual representation
        eigen_line = Line(start=axes.c2p(-3, 0), end=axes.c2p(3, 0), color="#FFFF00")
        self.play(Create(eigen_line), run_time=1)

        # === Animation for Lecture Line 4 ===
        self.lecture[3].set_color("#FFFF00")
        both = VGroup(vec_rot, vec_stretch)
        self.play(Indicate(both), run_time=1)

        # === Animation for Lecture Line 5 ===
        self.lecture[4].set_color("#00FF00")
        self.play(axes.animate.apply_matrix([[2, 0], [0, 1]]), run_time=2)
        self.wait(1)
