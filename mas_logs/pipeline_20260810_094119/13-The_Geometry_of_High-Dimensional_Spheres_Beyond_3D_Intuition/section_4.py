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
        self.setup_layout("Practical Application: High-Dimensional Data", [
            "Data in high dimensions hides on surfaces.",
            "Curse of dimensionality affects distance metrics.",
            "Distance resembles navigation on a hypersphere."
        ])
        
        # === Animation for Lecture Line 1 ===
        # Represent data points as dots in high-D using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg].
        sphere_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/sphere.svg")
        dots = VGroup(*[Dot(color=BLUE_B).move_to(np.random.normal(0, 0.5, 3)) for _ in range(30)])
        visual_group = VGroup(sphere_icon, dots).arrange(DOWN)
        self.place_in_area(visual_group, 'B3', 'E6', scale_factor=0.6)
        self.play(FadeIn(visual_group), self.lecture[0].animate.set_color(BLUE_B))
        self.wait(1)

        # === Animation for Lecture Line 2 ===
        # Highlight sparsity in high-dimensional space.
        self.play(visual_group.animate.scale(0.8).set_color(BLUE_D), self.lecture[1].animate.set_color(YELLOW))
        self.wait(1)

        # === Animation for Lecture Line 3 ===
        # Contrast nearest-neighbor distance with average distance using [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg].
        compass_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/compass.svg")
        hypersphere = Circle(radius=1.0, color=GREEN_C)
        combined_visual_group = VGroup(hypersphere, compass_icon).arrange(RIGHT)
        self.place_in_area(combined_visual_group, 'B2', 'E4', scale_factor=0.5)
        self.play(Create(combined_visual_group), self.lecture[2].animate.set_color(GREEN_C))
        self.wait(2)
