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
        self.setup_layout("The Concept of a Basis", [
            "A Basis is a minimalist spanning set.",
            "Basis vectors must be linearly independent.",
            "Every point has unique basis coordinates."
        ])

        # Setup Axes/Grid area
        axes = Axes(x_range=[-3, 3], y_range=[-3, 3], axis_config={"include_tip": True}).scale(0.5)
        self.place_in_area(axes, "C2", "F5", scale_factor=0.6)

        # === Animation for Lecture Line 1 ===
        self.lecture[0].set_color("#00FF7F")
        v1 = Vector([1, 1], color="#00FF7F")
        v2 = Vector([1, -0.5], color="#00FF7F")
        basis_group = VGroup(v1, v2).move_to(axes.get_origin())
        self.play(Create(basis_group))

        # === Animation for Lecture Line 2 ===
        self.lecture[1].set_color("#00FF7F")
        # Load asset: [Asset: /scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg]
        grid_icon = SVGMobject("/scratch/pawsey1357/jthen/Code2Video/assets/icon/grid.svg")
        self.place_at_grid(grid_icon, "B3", scale_factor=0.5)
        self.play(FadeIn(grid_icon), run_time=1.5)

        # === Animation for Lecture Line 3 ===
        self.lecture[2].set_color("#00FF7F")
        w = Vector([1.5, 0.5], color=YELLOW)
        # Apply positioning requirement for yellow vector
        self.place_at_grid(w, "C2", scale_factor=0.7)
        self.play(Create(w))
        self.wait(1)
