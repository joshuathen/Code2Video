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
        self.setup_layout("Beyond Arrows: Diverse Vector Spaces", [
            "Vector spaces are surprisingly diverse structures.",
            "Functions act as vectors in function spaces.",
            "Adding functions mirrors geometric vector addition."
        ])
        
        # Define objects
        # Coordinate vector
        vec = Arrow(start=ORIGIN, end=RIGHT*1.5 + UP*1.5, color=WHITE)
        vec_label = Text("Vector", font_size=18).next_to(vec, UP, buff=0.1)
        vec_group = VGroup(vec, vec_label)
        
        # Function curve
        axes = Axes(x_range=[-1, 2], y_range=[-1, 2], axis_config={"include_tip": False}).scale(0.3)
        func_curve = axes.plot(lambda x: x**2, x_range=[0, 1.5], color=WHITE)
        func_label = Text("Function", font_size=18).next_to(func_curve, RIGHT, buff=0.1)
        func_group = VGroup(axes, func_curve, func_label)

        # Asset loading placeholders (asset path /scratch/pawsey1357/jthen/Code2Video/assets/icon/none.svg)
        # Using placeholder since none.svg effectively does nothing visual here
        
        # === Animation for Lecture Line 1 ===
        self.place_in_area(vec_group, 'B4', 'C6', scale_factor=0.6)
        self.place_in_area(func_group, 'D4', 'E6', scale_factor=0.6)
        self.play(Create(vec_group), Create(func_group))
        self.lecture[0].set_color("#FFFFFF")

        # === Animation for Lecture Line 2 ===
        highlight_rect = SurroundingRectangle(func_group, color="#FF8080", buff=0.1)
        self.play(Create(highlight_rect))
        self.wait(1)
        self.play(FadeOut(highlight_rect))
        self.lecture[1].set_color("#FF8080")

        # === Animation for Lecture Line 3 ===
        # Morphing: Creating a transition effect
        self.play(
            vec.animate.become(func_curve.copy()),
            FadeOut(vec_label),
            run_time=2
        )
        self.lecture[2].set_color("#80FF80")
        self.wait(2)
